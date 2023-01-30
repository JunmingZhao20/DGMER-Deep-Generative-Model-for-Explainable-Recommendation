from transformers import GPT2LMHeadModel, GPT2Config
import torch.nn as nn
import torch
import copy

class Prompt_Model:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        model.initiate()
        return model


    def initiate(self):
        '''
        初始化，设置MLP recommender
        '''
        config = GPT2Config()
        n_hidden = config.n_embd    #以最后一层的hidden_state作为输入， 因而需要n_hidden
        self.recommender = MLP(n_hidden)



    def forward(self, prompt, explanation, exp_mask, ignore_index=-100):
        '''
        Args:
            prompt: 输入的prompt
            explanation: 输入的explanation,在generation 阶段， 仅需输入<eos>
            exp_mask: explanation mask 注意：generation 阶段不需要 attention mask
            ignore_index: GPT2 label项所需， 根据官方文档，默认-100
        Returns:
            out : GPT2LMHeadModel输出，loss, logits, hidden_states
            rating:预测评分
        '''
        device = prompt.device
        src_len = prompt.shape[1]      # prompt句子的长度
        text = torch.cat([prompt, explanation], 1)  # 输入的句子，将prompt 和 explanation拼接 (batch_size, total_len)
        src = self.transformer.wte(text)  # (batch_size, total_len, emsize)


        if exp_mask is None:
            # auto-regressive generation
            out = super().forward(inputs_embeds=src, output_hidden_states= True)
        else:
            # training阶段

            #attention mask
            prompt_mask = torch.ones_like(prompt, dtype=torch.int64).to(device)    #prompt部分的mask
            pad_input = torch.cat([prompt_mask, exp_mask], 1)  # prompt部分的mask 和 explanation mask 进行拼接 (batch_size, total_len)

            # prediction for training
            pred_prompt = torch.full_like(prompt, ignore_index, dtype=torch.int64).to(device)  # 整个prompt 部分都不需要计算损失 (batch_size, src_len)
            pred_exp = torch.where(exp_mask == 1, explanation, torch.tensor(ignore_index).to(device))  # replace <pad> with ignore_index
            prediction = torch.cat([pred_prompt, pred_exp], 1)  # (batch_size, total_len)
            out = super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction, output_hidden_states= True)


        #对《Personalized Prompt Learning for Explainable Recommendation》文章的改进， 使用最后一层hidden_state， 位于prompt末尾的输出，作为MLP的输入
        r_input = out.hidden_states[12][:, src_len, :]    #(batch_size, emsize)
        rating = self.recommender(r_input)

        return out, rating


class PromptLearning(Prompt_Model, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MLP(nn.Module):
    '''
    MLP作为recommender， 为与原文形成对比， 层数，参数均与原文一致
    '''
    def __init__(self, emsize, hidden_size=400, num_layers=2):
        '''
        模型架构：4层：首尾+中间2层 (hidden_size, hidden_size)
        '''
        super(MLP, self).__init__()
        self.first_layer = nn.Linear(emsize, hidden_size)
        self.last_layer = nn.Linear(hidden_size, 1)
        layer = nn.Linear(hidden_size, hidden_size)
        self.layers = _get_clones(layer, num_layers)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.first_layer.weight.data.uniform_(-initrange, initrange)
        self.first_layer.bias.data.zero_()
        self.last_layer.weight.data.uniform_(-initrange, initrange)
        self.last_layer.bias.data.zero_()
        for layer in self.layers:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.zero_()

    def forward(self, input):  # (batch_size, emsize)
        hidden = self.sigmoid(self.first_layer(input))  # (batch_size, hidden_size)
        for layer in self.layers:
            hidden = self.sigmoid(layer(hidden))  # (batch_size, hidden_size)
        rating = torch.squeeze(self.last_layer(hidden))  # (batch_size,)
        return rating



class UIPrompt:
    '''
    原文模型，用于做对比
    '''
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, nuser, nitem, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        model.init_prompt(nuser, nitem)
        return model

    def init_prompt(self, nuser, nitem):
        self.src_len = 2
        config = GPT2Config()
        emsize = config.n_embd  #获取GPT2的n_hidden
        #user, item过embedding, 即原文中的look_up matrix
        self.user_embedding = nn.Embedding(nuser, emsize)
        self.item_embedding = nn.Embedding(nitem, emsize)

        self.recommender = MLP_for_comp(emsize)

        #初始化
        initrange = 0.1
        self.user_embedding.weight.data.uniform_(-initrange, initrange)
        self.item_embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, user, item, text, mask, ignore_index=-100):
        device = user.device
        batch_size = user.size(0)
        embedded_user = self.user_embedding(user)
        embedded_item = self.item_embedding(item)
        embedded_text = self.transformer.wte(text)  # (batch_size, tgt_len, emsize)

        src_prompt = torch.cat([embedded_user.unsqueeze(1), embedded_item.unsqueeze(1), embedded_text], 1)  # (batch_size, total_len, emsize)

        #MLP的输入是user_embedding和item_embedding
        rating = self.recommender(embedded_user, embedded_item)

        if mask is None:
            # auto-regressive generation
            return super().forward(inputs_embeds=src_prompt), rating
        else:
            # training阶段

            # attention mask
            prompt_mask = torch.ones((batch_size, self.src_len), dtype=torch.int64).to(device)
            total_mask = torch.cat([prompt_mask, mask], 1)  # (batch_size, total_len)

            # prediction for training
            pred_prompt = torch.full((batch_size, self.src_len), ignore_index, dtype=torch.int64).to(device)  # (batch_size, src_len)
            pred_exp = torch.where(mask == 1, text, torch.tensor(ignore_index).to(device))  # replace <pad> with ignore_index
            prediction = torch.cat([pred_prompt, pred_exp], 1)  # (batch_size, total_len)

            return super().forward(attention_mask=total_mask, inputs_embeds=src_prompt, labels=prediction), rating


class PELER(UIPrompt, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)


class MLP_for_comp(nn.Module):
    '''
    原文模型中用到的MLP recommender
    '''
    def __init__(self, emsize, hidden_size=200, num_layers=2):
        '''
        输入user_embedding和item_embedding,二者拼接进入MLP， MLP架构：4层：首尾+中间2层 (hidden_size, hidden_size)
        Args:
            emsize: user_embedding和item_embedding 的size
            hidden_size: 中间层大小
            num_layers: 中间层数，default = 2
        '''
        super(MLP_for_comp, self).__init__()
        self.first_layer = nn.Linear(emsize*2, hidden_size)
        self.last_layer = nn.Linear(hidden_size, 1)
        layer = nn.Linear(hidden_size, hidden_size)
        self.layers = _get_clones(layer, num_layers)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.first_layer.weight.data.uniform_(-initrange, initrange)
        self.first_layer.bias.data.zero_()
        self.last_layer.weight.data.uniform_(-initrange, initrange)
        self.last_layer.bias.data.zero_()
        for layer in self.layers:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.zero_()

    def forward(self, user, item):
        ui_cat = torch.cat([user, item], 1)  # (batch_size, emsize * 2)
        hidden = self.sigmoid(self.first_layer(ui_cat))
        for layer in self.layers:
            hidden = self.sigmoid(layer(hidden))
        rating = torch.squeeze(self.last_layer(hidden))
        return rating