import os
import math
import torch
import torch.nn as nn
import argparse
from transformers import AdamW, BertTokenizer
from module import PromptLearning
from Prepare_data import DataLoader, Batchify, now_time, ids2tokens
from metrics import rouge_score, bleu_score, unique_sentence_percent, feature_diversity, feature_detect, feature_matching_ratio, feature_coverage_ratio, mean_absolute_error, root_mean_square_error, unique_sentence_percent


# ----------------以下设置超参数-----------------

#超参数写法参照了https://github.com/lileipisces/PEPLER，方便将结果与其对比
parser = argparse.ArgumentParser(description='GPT2_improved model by Junming Zhao')
parser.add_argument('--data_path', type=str, default=None,
                    help='path for loading the csv data')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate for the model')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--checkpoint', type=str, default='./pepler/',
                    help='directory to save the final model')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--endure_times', type=int, default=5,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--words', type=int, default=20,
                    help='number of words to generate for each sample')
parser.add_argument('--rating_reg', type=float, default=0.01,
                    help='regularization on recommendation task')
parser.add_argument('--text_reg', type=float, default=1.0,
                    help='regularization on text generation task')
args = parser.parse_args()

device = torch.device('cuda' if args.cuda else 'cpu')   #设置训练设备，优先GPU
#指定checkpoint文件夹
if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)
model_path = os.path.join(args.checkpoint, 'model.pt')
prediction_path = os.path.join(args.checkpoint, args.outf)



#---------------------------以下加载数据，准备模型-----------------------------

#由于中文gpt2没有对应的tokenizer, 参照官网说明，使用 BertTokenizer, uer/gpt2-chinese-cluecorpussmall
tokenizer = BertTokenizer.from_pretrained('uer/gpt2-chinese-cluecorpussmall', bos_token='<bos>', eos_token='<eos>', pad_token='<pad>')
corpus = DataLoader(args.data_path, tokenizer, args.words)
feature_set = corpus.feature_set
train_data = Batchify(corpus.train,  tokenizer, '<bos>', '<eos>', args.words, args.batch_size, shuffle=True)
val_data = Batchify(corpus.valid,  tokenizer, '<bos>', '<eos>', args.words, args.batch_size)
test_data = Batchify(corpus.test,  tokenizer, '<bos>', '<eos>', args.words, args.batch_size)

nuser, nitem, ntoken = len(corpus.user), len(corpus.item), len(tokenizer)
model = PromptLearning.from_pretrained('uer/gpt2-chinese-cluecorpussmall')  #中文GPT2预训练语言模型uer/gpt2-chinese-cluecorpussmall
model.resize_token_embeddings(ntoken)
model.to(device)
optimizer = AdamW(model.parameters(), lr=args.lr)
rating_criterion = nn.MSELoss()


#-------------------------以下 train, evaluate, generate函数-------------------------

def train(data):
    model.train()
    text_loss = 0.
    rating_loss = 0.
    total_sample = 0
    while True:
        explan, mask, prompt, rating = data.next_batch()
        explan = explan.to(device)
        rating = rating.to(device)
        mask = mask.to(device)
        prompt = prompt.to(device)
        optimizer.zero_grad()
        outputs, rating_pred = model(prompt, explan, mask)
        #使用两种loss:t_loss:由GPT2计算得文本损失， r_loss：预测rating与真实rating MSE
        t_loss = outputs.loss
        r_loss = rating_criterion(rating_pred, rating)
        loss = args.text_reg * t_loss + args.rating_reg * r_loss    #加权组成总损失，rating_reg=0.01， text_reg=1
        loss.backward()
        optimizer.step()

        batch_size = explan.size(0)
        #计算batch级别的损失
        text_loss += batch_size * t_loss.item()
        rating_loss += batch_size * r_loss.item()
        total_sample += batch_size

        #每200step进行一次报告
        if data.step % args.log_interval == 0 or data.step == data.total_step:
            cur_t_loss = text_loss / total_sample
            cur_r_loss = rating_loss / total_sample
            print(now_time() + 'text loss {:4.4f} | rating loss {:4.4f} | {:5d}/{:5d} batches'.format(
                math.exp(cur_t_loss), cur_r_loss, data.step, data.total_step))
            text_loss = 0.
            rating_loss = 0.
            total_sample = 0
        if data.step == data.total_step:
            break



def evaluate(data):
    model.eval()
    text_loss = 0.
    rating_loss = 0.
    total_sample = 0
    with torch.no_grad():
        while True:
            explan, mask, prompt, rating = data.next_batch()
            explan = explan.to(device)
            rating = rating.to(device)
            mask = mask.to(device)
            prompt = prompt.to(device)
            outputs, rating_pred = model(prompt, explan, mask)
            batch_size = explan.size(0)
            #与train时一致，loss:t_loss:由GPT2计算得文本损失， r_loss：预测rating与真实rating MSE
            t_loss = outputs.loss
            r_loss = rating_criterion(rating_pred, rating)
            text_loss += batch_size * t_loss.item()
            rating_loss += batch_size * r_loss.item()
            total_sample += batch_size

            if data.step == data.total_step:
                break
    return text_loss / total_sample, rating_loss / total_sample


def generate(data):
    model.eval()
    idss_predict = []
    input_features = []
    rating_predict = []
    with torch.no_grad():
        while True:
            #generate阶段，只需要输入prompt，无需再进行mask
            explan, _, prompt, _ = data.next_batch()
            prompt = prompt.to(device)
            text = explan[:, :1].to(device)  #不再输入整个explanation， 而是输入<eos>进行文本生成
            seq_len = explan.size(1)    #需要生成的句子长度
            for idx in range(seq_len):
                if idx == 0:
                    #首次，需要recommender rating
                    outputs, rating_pred = model(prompt, text, None)
                    rating_predict.extend(rating_pred.tolist())
                else:
                    #非首次，不需要需要recommender rating
                    outputs, _ = model(prompt, text, None)
                # 自回归：一次生成一个词
                last_token = outputs.logits[:, -1, :]  # (batch_size, ntoken)
                word_prob = torch.softmax(last_token, dim=-1)
                token = torch.argmax(word_prob, dim=1, keepdim=True)  #选择最大概率的词
                text = torch.cat([text, token], 1)
            ids = text[:, 1:].tolist()  # remove bos
            idss_predict.extend(ids)
            input_features.extend(prompt.tolist()[2:])  #与原文不同，原文并不将user, item作为prompt一部分，此处因为prompt使用了user, item，故feature是prompt[2:]

            if data.step == data.total_step:
                break
    return idss_predict, input_features, rating_predict


#----------------以下进行模型训练与评估--------------

#模型训练
print('-' * 30+'以下是训练过程'+'-' * 30)
min_loss = float('inf')     #记录最小损失
endure = 0                  #endure次数记录
for epoch in range(1, args.epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    train(train_data)
    val_t_loss, val_r_loss = evaluate(val_data)
    val_loss = val_t_loss + val_r_loss
    print(now_time() + 'text loss {:4.4f} | rating loss {:4.4f} | valid loss {:4.4f} on validation'.format(
        math.exp(val_t_loss), val_r_loss, val_loss))
    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < min_loss:
        #当validation loss还在下降时，继续训练，该训练策略参照原文
        min_loss = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model, f)
    else:
        endure += 1
        print(now_time() + 'Endured {} time(s)'.format(endure))
        if endure == args.endure_times:
            print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
            break

#模型评估
#载入最终训练好的模型
with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)
test_t_loss, test_r_loss = evaluate(test_data)
print('-' * 30+'以下是在测试集上的表现'+'-' * 30)
print(now_time() + 'text loss {:4.4f} | rating loss {:4.4f} on test data'.format(math.exp(test_t_loss), test_r_loss))
print(now_time() + 'Generating text')
idss_predicted, features, rating_pred = generate(test_data)
#各metric与原文一样，方便对比
predicted_rating = [(r, p) for (r, p) in zip(test_data.rating.tolist(), rating_pred)]
RMSE = root_mean_square_error(predicted_rating, corpus.max_rating, corpus.min_rating)
print(now_time() + 'RMSE {:7.4f}'.format(RMSE))
MAE = mean_absolute_error(predicted_rating, corpus.max_rating, corpus.min_rating)
print(now_time() + 'MAE {:7.4f}'.format(MAE))
tokens_test = [ids2tokens(ids[1:], tokenizer, '<eos>') for ids in test_data.seq.tolist()]
tokens_predict = [ids2tokens(ids, tokenizer, '<eos>') for ids in idss_predicted]
BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
print(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
print(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))
USR, USN = unique_sentence_percent(tokens_predict)
print(now_time() + 'USR {:7.4f} | USN {:7}'.format(USR, USN))
feature_batch = feature_detect(tokens_predict, feature_set)
DIV = feature_diversity(feature_batch)  # time-consuming
print(now_time() + 'DIV {:7.4f}'.format(DIV))
FCR = feature_coverage_ratio(feature_batch, feature_set)
print(now_time() + 'FCR {:7.4f}'.format(FCR))
FMR = feature_matching_ratio(feature_batch, test_data.feature)
print(now_time() + 'FMR {:7.4f}'.format(FMR))
text_test = [' '.join(tokens) for tokens in tokens_test]
text_predict = [' '.join(tokens) for tokens in tokens_predict]
ROUGE = rouge_score(text_test, text_predict)  # a dictionary
for (k, v) in ROUGE.items():
    print(now_time() + '{} {:7.4f}'.format(k, v))
text_out = ''
for (u, i, real, fake, r_real, r_pred) in zip(test_data.user, test_data.item, text_test, text_predict, test_data.rating.tolist(), rating_pred):
    text_out += '用户：{}\t电影：{}\n 用户评论：{}\n生成的解释：{}\n真实评分：{}\t预测评分{}\n\n'.format(corpus.user.idx2entity[u],  corpus.item.idx2entity[i], real, fake, r_real, r_pred)
with open(prediction_path, 'w', encoding='utf-8') as f:
    f.write(text_out)
print(now_time() + 'Generated text saved to ({})'.format(prediction_path))


