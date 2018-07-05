import os, sys
module_path = os.path.abspath(os.path.join('/home/ubuntu/fastai'))
if module_path not in sys.path: sys.path.append(module_path)
from fastai.nlp import *
from fastai.text import *
from fastai.conv_learner import *
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.svm import LinearSVC
import torch, pandas as pd, json, psutil, gc
from torchtext import vocab, data, datasets
from IPython import display
from IPython.display import clear_output
from numbers import Number
from enum import Enum, IntEnum

ETY_PATH = '/home/ubuntu/git/courses/deeplearning2/data/datasets/etymologies/'
MODELS_PATH = ETY_PATH + 'pytorchmodels/'

def log_progress(sequence, every=None, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )

        
        
##############################################
##############################################
################## DATA PREP #################
##############################################
##############################################

#### REVISION 4 separated out "." for ". : "
#### REVISION 3 (added [] to re_punc)
#Next revisision needs to get rid of 'to bat' issues keep apostrophes in the middle of words

def parseEtymology(etymology): 
    re_apos = re.compile("(\w)'s\b")         # make 's a separate word
    re_punc = re.compile("([“”,;:_?!—\+\/\[\]])") # add spaces around punctuation include key_word
    re_periods = re.compile("(\.)(?! : )") # add spaces around periods (but not on key words)
    re_front_paren = re.compile("(?<= )(\()(?=[^ ])") # add spaces around front parens (try to avoid key words)
    re_back_paren = re.compile("(\( .+?)(?<=[^ ])(\))(?= |$|.)")
    re_front_dbl_quote = re.compile("(?<=[ (])(\")(?=[^ ])") # add spaces around front double quotes 
    re_back_dbl_quote = re.compile("(?<=[^ ])(\")(?=[ )])") # add spaces around back double quotes 
    re_plus = re.compile("\\\\\+") # replace \\+ with +
    re_mult_space = re.compile("  +")        # replace multiple spaces with just one
    # re_front_single_quote = re.compile('(?<!Tae)(?<! )(\' )') # find single quote with trailing space
    # re_back_single_quote = re.compile('( \')(?! )')# find single quote with leading space

    etymology = etymology.replace('*', '') #remove the asterisks
    etymology = etymology.replace('\u200e', '') #remove the "left-to-right" markers
    etymology = re_plus.sub(r"+", etymology) # replace \\+ with +
    etymology = re_periods.sub(r" \1 ", etymology) #add spaces around periods (avoid key words)
    etymology = re_punc.sub(r" \1 ", etymology) #add spaces around puncuation
#     etymology = re_front_single_quote.sub(r" \1 ", etymology) #add spaces around single quotes
#     etymology = re_back_single_quote.sub(r" \1 ", etymology) #add spaces around single quotes

    while True:
        start_len = len(etymology)
        etymology = re_front_paren.sub(r"\1 ", etymology)
        etymology = re_back_paren.sub(r"\1 \2 ", etymology)
        etymology = re_front_dbl_quote.sub(r" \1 ", etymology)
        etymology = re_back_dbl_quote.sub(r" \1 ", etymology)
        if start_len == len(etymology): break
    etymology = re_mult_space.sub(' ', etymology) #Reduce duplicate spaces
    return etymology.strip() #remove trailing spaces

def add_lang_to_codes(row, languages, max_lang_len=4):
    if row['code_source'] in ['none', 'broken']: return row['code']
    
    words = row['parsed_etymology'].split(' '); words
    codes = row['code']; codes
    if type(codes) == str: codes = eval(codes)

    for start in range(len(words)):
        for length in reversed(range(1, max_lang_len + 1)):
            if start+length>len(codes): continue
            if not all([code == 'None' for code in codes[start:start+length]]): continue #continue if not all 'None'

            if ' '.join(words[start:start+length]) in languages:
                codes[start:start+length] = ['lang'] * length
    return codes


def pad_sequence(array, maxlen, value=0): return array[:maxlen] + [value] *(maxlen - len(array))

def make_char_codes_col(row, code_vocab, maxlen):
    split_words = row['parsed_etymology'].split(' '); codes = row['code']
    if type(codes) == str: codes = eval(codes)
    try:
        chars_codes = [[codes[i]] * len(e) + ['None'] for i,e in enumerate(split_words)]
        chars_codes = [item for sublist in chars_codes for item in sublist][:-1]
        chars_codes = [code_vocab.index(str(c)) if str(c) in code_vocab else len(code_vocab)-1 for c in chars_codes]
    except Exception as e:
        print(row, e)
        raise e
    return pad_sequence(chars_codes, maxlen)

def make_chars_column(row, char_vocab, maxlen):
    char_list = list(row['parsed_etymology'])
    chars = [char_vocab.index(c) if c in char_vocab else len(char_vocab)-1 for c in char_list]
    return pad_sequence(chars, maxlen)

def make_model_data(model_data, char_vocab, code_vocab, train=False, maxlen = 300, name = None):
    """
    train=True loads chars and codes and only where the code is available
    train=False loads only chars for all data
    """
    if not name: name = 'all_data'
    drop_col = model_data.columns.tolist()
    
    if train:
        model_data = model_data[(model_data["code_source"] != 'none') & (model_data["code_source"] != 'broken')]
        model_data['char_codes'] = model_data.apply(lambda x: make_char_codes_col(x, code_vocab, maxlen), axis=1)
        if 'char_codes' in drop_col: drop_col.remove('char_codes') # Only drop for training (only training has it)
        if not name: name = 'training_data'
        
    model_data['chars'] = model_data.apply(lambda x: make_chars_column(x, char_vocab, maxlen), axis=1)
    if 'chars' in drop_col: drop_col.remove('chars') # Drop if it is in the list
    
    return model_data.drop(columns=drop_col)

def load_table(table_name = 'full_processed_wiktionary_pages_current.csv', index_col=0):
    pd.set_option('chained', None)
    converters = {'word': str, 'language': str, 'etymology': str} # To prevent reading strings as floats
    return pd.read_csv(ETY_PATH + table_name, low_memory=False, index_col=index_col, converters=converters)

def save_table(table, table_name):
    if 'temp' in table.columns: 
        col_to_remove = ['temp']
        table = table.drop(columns=col_to_remove)
    return table.to_csv(ETY_PATH + table_name, index=True)

def load_char_embeds(file_name):
    """ Returns: char_vocab, char_embeddings_vectors """
    char_embeddings_path = ETY_PATH + file_name
    char_embeddings_df = pd.read_csv(char_embeddings_path, low_memory=False)
    char_embeddings_vectors = char_embeddings_df.drop(columns='char').values
    char_vocab = char_embeddings_df['char'].tolist()
    return char_vocab, char_embeddings_vectors 

def load_code_vocab(file_name, code_vocab_size):
    """ Returns: code_vocab """
    code_vocab_path = ETY_PATH + file_name
    code_vocab_df = pd.read_csv(code_vocab_path, low_memory=False)
    code_vocab = code_vocab_df['code'].tolist()[:code_vocab_size]
    return code_vocab

def save_training_data(training_data, file_name):
    col_to_remove = list(training_data.columns)
    col_to_remove.remove('chars'); col_to_remove.remove('char_codes') #remove from "columns to remove" 
    return training_data.drop(columns=col_to_remove).to_pickle(ETY_PATH + file_name)
    
def load_training_data(file_name):
    return pd.read_pickle(ETY_PATH + file_name)
  
class EtyDataset(Dataset):
    def __init__(self, df, y_one_hot=False, y_vocab_size=4): 
        self.x = np.array(df['chars'].tolist())
        self.y = np.array(df['char_codes'].tolist())
        self.df, self.y_one_hot, self.y_vocab_size = df, y_one_hot, y_vocab_size
        self.shape = [len(self.df), len(self.df.iloc[0, 0])]
    def __getitem__(self, idx): 
        x = self.x[idx]; y = self.y[idx]
        if self.y_one_hot:            
            y_onehot = np.zeros((y.size, self.y_vocab_size))
            y_onehot[np.arange(y.size), y] = 1
            y = y_onehot #change y
        return {'x': x, 'y': y, 'i': self.df.iloc[idx].name}
        
    def __len__(self): return self.shape[0]
    
class PredDataset(Dataset):
    def __init__(self, x): self.x = x
    def __getitem__(self, idx): return A(self.x[idx], [2 for _ in self.x[idx]])
    def __len__(self): return len(self.x)
    
def seq2seq_loss(_input, target):
    sl,bs = target.size()
    sl_in,bs_in,nc = _input.size()
    if sl>sl_in: _input = F.pad(_input, (0,0,0,0,0,sl-sl_in))
    _input = _input[:sl]
    return F.cross_entropy(_input.view(-1,nc), target.view(-1))#, ignore_index=1)

def wd_acc(preds, targs): 
    return (torch.max(preds, dim=2)[1] == targs).float().mean()

def sent_acc(preds, targs): 
    return (torch.max(preds, dim=2)[1] == targs).min(0)[0].float().mean()
    
##############################################
##############################################
################## TRAINING ##################
##############################################
##############################################
    
def train_model(model, train_loader, test_loader, 
                loss_function, optimizer, n_classes,
                num_epochs = 1, update_freq = 50, break_on_val_inc=-1): 
    #loss graph is dictionary of trainings
        #Each training item contains the loss data, the epoch loss, the size of batch, epoch, and training_data
        
    size = round(train_loader.dataset.shape[0]/train_loader.batch_size)
    
    #memory test
    list(enumerate(train_loader))
    
    
    #Get all model data
    lr_data = model.loss_graph['lr_data']
    loss_data = model.loss_graph['loss_data']
    loss_data_indices = model.loss_graph['loss_data_indices']    
    val_loss_data = model.loss_graph['val_loss_data']
    val_loss_data_indices = model.loss_graph['val_loss_data_indices']
    loss_index = max(loss_data_indices + [0])
    
    train_data_size = train_loader.dataset.shape[0]
    
    prev_vals = []
    
    try:
        for epoch in range(num_epochs):

            for i, datum in log_progress(enumerate(train_loader), every=1, size=size):
                chars = V(datum['x']); codes = V(datum['y'])

                model.zero_grad()
                predicted_codes = model(chars)
                loss = loss_function(predicted_codes.view([-1, n_classes]), codes.view([-1]))
                del predicted_codes
                
                loss_index += len(chars)
                loss_data_indices.append(loss_index)
                loss_data.append(loss.data[0])
                
                lr_data.append(optimizer.param_groups[0]['lr'])
                
                loss.backward()
                optimizer.step()

                if (i+1) % update_freq == 0:
                    print ('Epoch [%d/%d], Iter [%d/%d] Training Loss: %.4f' 
                           %(epoch+1, num_epochs, i+1, len(train_loader.dataset)//train_loader.batch_size, loss.data[0]))

            #Validation
            val_loss = 0
            for i, datum in enumerate(test_loader):
                chars = Variable(datum['x']); codes = Variable(datum['y'])
                predicted_codes = model(chars)
        
                val_loss += loss_function(predicted_codes.view([-1, n_classes]), codes.view([-1])).data[0]
                del predicted_codes
            val_loss /= (i + 1) #must scale
            
            val_loss_data_indices.append(loss_index)
            val_loss_data.append(val_loss)
            
            print ('Epoch [%d/%d], Validation Loss: %.4f' % (epoch+1, num_epochs, val_loss))
            
            #If validation loss is not smaller than the 4th smallest loss,break
            if break_on_val_inc >= 0:
                prev_vals.append(val_loss)
                if val_loss > sorted(prev_vals)[min(len(prev_vals)-1,break_on_val_inc)]: 
                    print('Val loss increase, breaking')
                    return

    except KeyboardInterrupt as k:
        print('Keyboard Interrupt')
    except Exception as e:
        print('Error', e)
        raise Exception(e)
        
class LROptimizer(optim.Adam):
    def __init__(self, params, loader, num_epochs, init_lr = 1e-10, end_lr = 1):
        self.init_lr = init_lr
        self.end_lr = end_lr
        self.lr_increments = loader.dataset.shape[0] / loader.batch_size * num_epochs
        self.lr_growth_rate = self.end_lr /(self.init_lr) ** (1/self.lr_increments)
        super(LROptimizer, self).__init__(params, lr = self.init_lr)

    def step(self):
        for group in self.param_groups:
            group['lr'] *= self.lr_growth_rate
        return super().step()

def learn_lr(model, loader, loss_func, n_classes, 
             num_epochs=1, init_lr = 1e-10, end_lr = 1):
    
    size = round(loader.dataset.shape[0]/loader.batch_size)
    
    train_data_size = train_loader.dataset.shape[0]
    optimizer = LROptimizer(model.parameters(), loader, num_epochs, init_lr=init_lr, end_lr=end_lr)
    init_loss = None
    losses = []
    lrs = []
    
    plt.xscale('log')
    plt.yscale('log')

    try:
        for epoch in range(num_epochs):
            
            for i, datum in enumerate(train_loader):
                chars = Variable(datum['x']); codes = Variable(datum['y'])
                
                model.zero_grad()
                predicted_codes = model(chars)
                loss = loss_func(predicted_codes.view([-1, n_classes]), codes.view([-1]))

                if not init_loss: 
                    init_loss = loss.data[0]
                elif loss.data[0] > init_loss + 1: raise Exception('loss too high')

                losses.append(loss.data[0])
                cur_lr = optimizer.param_groups[0]['lr']
                lrs.append(cur_lr)

                plt.plot([cur_lr], [loss.data[0]], 'bo', markersize=3)
                display.clear_output(wait=True)
                display.display(plt.gcf())
                
                loss.backward()
                optimizer.step()
        display.clear_output(wait=True)
    except Exception as e:
        display.clear_output(wait=True)
        raise(e)
        print(e)
    finally:
        plt.grid()

def plot_losses(loss_dict, height=0.06, start_x=0, width=None, plot=None):
    plt.figure(num=None, figsize=(15, 10), dpi=90, facecolor='w', edgecolor='k')
    max_x = 0
    for name, lg in loss_dict.items():
        
        loss_data_indices = lg['loss_data_indices']
        loss_data = lg['loss_data']
        val_loss_data_indices = lg['val_loss_data_indices']
        val_loss_data = lg['val_loss_data']
        
        max_x = max(max_x, max(lg['loss_data_indices']))
        
        if plot == 'epoch':
            epoch_loss_indices = [i for i, index in enumerate(loss_data_indices) if index in val_loss_data_indices]
            loss_data = [loss for i, loss in enumerate(loss_data) if i in epoch_loss_indices]
            loss_data_indices = [index for index in loss_data_indices if index in val_loss_data_indices]

        if plot != 'val':
            plt.plot(loss_data_indices, loss_data, label=name+'_loss')

        plt.plot(val_loss_data_indices, val_loss_data, label=name+'_val_loss')

        plt.grid()
        
    if not width:
        width =max_x+max_x/100
    
    plt.axis([start_x, width, 0, height])
    plt.legend()
    
##############################################
##############################################
################### MODEL TIRAMISU ###########
##############################################
##############################################
    
def create_tiramisu_emb(emb_mat, non_trainable=False):
    """ Provide embedding matrix (emb_mat)
    returns -
        emb: nn.Embedding(num_embeddings, emb_size)
        emb_dim: size of the embedding (100 for chars)
        num_embeddings: number of embeddings (total num of words in this case)
    """ 
    if type(emb_mat)==np.ndarray:
        emb_mat = FloatTensor(emb_mat).cuda()
    num_embeddings, emb_dim = emb_mat.size() #Returns the dimensions of the embedding
    emb = nn.Embedding(num_embeddings, emb_dim).cuda()
    emb.weight = nn.Parameter(emb_mat)
    if non_trainable:
        for param in emb.parameters(): 
            param.requires_grad = False
    return emb, emb_dim, num_embeddings

class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(in_channels, growth_rate, kernel_size=(11,1), stride=1, padding=(5,0), bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))

    def forward(self, x):
        return super().forward(x)
    
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer( #layers are (48,12)(60,12)(72,12)(84,12)(96,12)
            in_channels + i*growth_rate, growth_rate) #growth_rate = 12
            for i in range(n_layers)]) #n_layers = (5,5,5,5,5)

    def forward(self, x):
        if self.upsample:
            new_features = []
            #we pass all previous activations into each dense layer normally
            #But we only store each dense layer's output in the new_features array
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features,1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1) # 1 = channel axis
            return x
        
class TransitionDown(nn.Sequential):
    def __init__(self, in_channels, contract_rate=2):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, in_channels,
                                          kernel_size=1, stride=1,
                                          padding=0, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))
        self.add_module('maxpool', nn.MaxPool2d((contract_rate,1)))

    def forward(self, x):
        return super().forward(x)

class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels, contract_rate=2):
        super().__init__()
        self.convTrans = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3,1), stride=contract_rate, padding=0, bias=True)

    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out
    
class Bottleneck(nn.Sequential):
    def __init__(self, in_channels, growth_rate, n_layers):
        super().__init__()
        self.add_module('bottleneck', DenseBlock(
            in_channels, growth_rate, n_layers, upsample=True))

    def forward(self, x):
        return super().forward(x)

def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]

class FCDenseNet(nn.Module):
    def __init__(self, emb_mat, in_channels=1, down_blocks=(5,5,5,5,5), #in_channels = 3
                 up_blocks=(5,5,5,5,5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, n_classes=4):
        super().__init__()
        self.loss_graph = {'lr_data': [],
                           'loss_data': [],
                           'loss_data_indices': [],
                           'val_loss_data': [],
                           'val_loss_data_indices': []}
        self.n_classes = n_classes
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []

        ## Embedding ##
        emb, _, _ = create_tiramisu_emb(emb_mat)
        self.add_module('embedding', emb)
        
        ## First Convolution ##

        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                  out_channels=out_chans_first_conv, kernel_size=(3, 100),
                  stride=1, padding=(1,0), bias=True))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate*down_blocks[i])
            skip_connection_channel_counts.insert(0, cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        #####################
        #     Bottleneck    #
        #####################

        self.add_module('bottleneck',Bottleneck(cur_channels_count,
                                     growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate*bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks)-1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i],
                    upsample=True))
            prev_block_channels = growth_rate*up_blocks[i]
            cur_channels_count += prev_block_channels

        ## Final DenseBlock ##

        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
                upsample=False))
        cur_channels_count += growth_rate*up_blocks[-1]

        ## Softmax ##

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
               out_channels=n_classes, kernel_size=(1), stride=1,
                   padding=0, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        bs, h, w = x.size()
        x = x.view(bs, 1, h, w)
        out = self.firstconv(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        out = self.softmax(out)
        out = out.squeeze(3).permute(0,2,1).contiguous()
        return out


    
####### SAVING ######
def get_local_time():
    return time.strftime("%m-%d-%y_%H-%M-%S", time.localtime(time.time()-60*60*4))

def save_model_func(model, base_name='', model_name = None):
    if not model_name:
        model_name = base_name + '_model_{}'.format(get_local_time())
    torch.save(model, MODELS_PATH+model_name)

def load_model_func(model_name):
    return torch.load(MODELS_PATH+model_name)

##############################################
##############################################
############ MODEL POS_CLASSIFIER ############
##############################################
##############################################

class POSClassifier(nn.Module):
    def __init__(self, layers, drops):
        super().__init__()
        self.layers = nn.ModuleList([
            LinearBlockSentence(layers[i], layers[i + 1], drops[i]) for i in range(len(layers) - 1)])

    def pool(self, x, bs, is_max):
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(1,2,0), (1,)).view(bs,-1)

    def forward(self, input):
        raw_outputs, outputs = input #Can try to switch these (using hidden state if outputs is second)
        output = outputs[-1]
        sl,bs,_ = output.size() #(~70, 6, 400) sl is sentence length, _ is the rnn dim
#         avgpool = self.pool(output, bs, False)
#         mxpool = self.pool(output, bs, True)
#         x = torch.cat([output[-1], mxpool, avgpool], 1)
        x = output
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)
        return l_x, raw_outputs, outputs
    
class LinearBlockSentence(nn.Module):
    def __init__(self, ni, nf, drop):
        super().__init__()
        self.lin = nn.Linear(ni, nf)
        self.drop = nn.Dropout(drop)
        self.bn = nn.BatchNorm1d(ni)

    def forward(self, x):
        o1 = self.bn(x.permute(1,2,0).contiguous())
        o2 = self.drop(o1)
        return self.lin(o2.permute(0,2,1))
    
def get_rnn_pos_classifer(bptt, max_seq, n_class, n_tok, emb_sz, n_hid, n_layers, pad_token, layers, drops, bidir=False,
                      dropouth=0.3, dropouti=0.5, dropoute=0.1, wdrop=0.5, qrnn=False):
    rnn_enc = MultiBatchRNN(bptt, max_seq, n_tok, emb_sz, n_hid, n_layers, pad_token=pad_token, bidir=bidir,
                      dropouth=dropouth, dropouti=dropouti, dropoute=dropoute, wdrop=wdrop, qrnn=qrnn)
    return SequentialRNN(rnn_enc, POSClassifier(layers, drops))


##############################################
##############################################
################## ANALYSIS ##################
##############################################
##############################################


# Displays the pred, actu, and sentence in a vertical format
# def show_analysis(pred, actu, sent):
def show_word_analysis(pred_table_loc):
    
    returnstring = 'index: %d' % pred_table_loc.name
    if 'pred_cor' in pred_table_loc: returnstring += '; pred_cor: ' + str(pred_table_loc['pred_cor'])
    if 'pred_conf' in pred_table_loc: returnstring += '; pred_conf = %.4f' % pred_table_loc['pred_conf']
    if 'pred_error' in pred_table_loc: returnstring += '; pred_error: ' + str(pred_table_loc['pred_error'])
    
    sent = pred_table_loc['parsed_etymology'].split(' ')
    
    if pred_table_loc['code'] == pred_table_loc['code']: #then it is NaN
        actu = pred_table_loc['code']
        if type(actu) == str: actu=eval(actu)
    else:
        actu = ['None']*len(sent)
        
    if 'pred_code' in pred_table_loc:
        pred = pred_table_loc['pred_code']
        if type(pred) == str: pred=eval(pred)
        color_col = pred
    elif 'pred_word_code' in pred_table_loc: raise Exception("predictions must be in column 'pred_code'")
    else: 
        pred = ['None']*len(sent)
        color_col = actu

    colored_sent = []
    for i, p in enumerate(color_col):
        if   p == 'key': colored_sent.append('\x1b[1;30m'+sent[i]+'\x1b[0m')
        elif p == 'lang': colored_sent.append('\x1b[96m'+sent[i]+'\x1b[0m')
        elif p[0] == '[': 
            if len(eval(p)) == 2: colored_sent.append('\x1b[1;94m'+sent[i]+p+'\x1b[0m')
            elif len(eval(p)) == 3: colored_sent.append('\x1b[1;95m'+sent[i]+p+'\x1b[0m')
            else: colored_sent.append('\x1b[1;31m'+sent[i]+p+'\x1b[0m')
        else: colored_sent.append(sent[i])

    returnstring += '\n' + ' '.join(colored_sent)
    header = '\n{:15}{:15}'.format('pred', 'word')
    
    if actu:
        header = '\n{:15}{:15}{:15}'.format('pred', 'actu', 'word')
    
    returnstring += header
    
    for j in range(len(list(sent))):
        w = sent[j]
        try:
            p = str(pred[j])
            if   p == 'key': w = '\x1b[1;30m' + w + '\x1b[0m'
            elif p == 'lang': w = '\x1b[96m' + w + '\x1b[0m'
            elif p[0] == '[': 
                if len(eval(p)) == 2: w = '\x1b[1;94m' + w + '\x1b[0m'
                elif len(eval(p)) == 3: w = '\x1b[1;95m' + w + '\x1b[0m'
                else: w = '\x1b[1;31m' + w + '\x1b[0m'
        except: pass
        
        if p == 'None': p = '-'
        nextline = '\n{:15}{:15}'.format(p,w)
        
        if actu:
            a = str(actu[j])
            if a == 'None': a = '-'
            nextline = '\n{:15}{:15}{:15}'.format(p,a,w)
        
        returnstring += nextline
        p = 'None' #defaults
        a = 'None'
    return returnstring

# Displays the pred, actu, and sentence in a vertical format
def show_analysis(pred, actu, sent):
    """provide 1-d array of ints for each variable"""
    
    if isinstance(sent[0], Number): #Convert to string if it is in numbers
        sent = ''.join([ety_chars_vocab[int(s)] for s in sent if s != char_vocab.index('<PAD>')]); sent
    else:
        sent = ''.join(sent)
    pred = [ety_codes_vocab[p] for p in pred]
    if actu: actu = [ety_codes_vocab[int(a)] for a in actu]
    
    returnstring = str(sent)
    returnstring += '\n{:5}{:10}{:10}'.format('word', 'pred', 'actu')
    for j in range(len(list(sent))):
        w = list(sent)[j]
        
        try:
            p = pred[j]
            if actu: 
                a = actu[j]
        except (KeyError, IndexError):
            pass
        
        else: a = 'None'
        if p == 'None': p = '-'
        if a == 'None': a = '-'
        returnstring += '\n{:5}{:10}{:10}'.format(w,p,a)

        p = 'None' #defaults
        a = 'None'
    return returnstring

def analyzePreds(actuals, predictions):
    """Provide count of correct and incorrect predictions"""
    combine = list(zip(actuals, predictions))
    if type(actuals) == np.ndarray:
        cor = [i for i,t in enumerate(combine) if all(t[0] == t[1])]
    else:
        cor = [i for i,t in enumerate(combine) if t[0] == t[1]]
    incor = [i for i in range(len(combine)) if i not in cor]
    return cor, incor

def getCharPredictions(model, loader, batch_limit=None):
    size = round(loader.dataset.shape[0]/loader.batch_size)-1 # Size for the log_progress bar
    if batch_limit: size = batch_limit
    length = loader.dataset.shape[1]
    
    pred_codes_raw = [] #= np.zeros([0, 300, code_vocab_size]) # Blank matrix to hold predictions
    actu_chars = np.zeros([0,length], dtype=int)
    actu_codes = np.zeros([0,length], dtype=int)
    actu_ids = np.zeros([0], dtype=int)

    for i, datum in log_progress(enumerate(loader), every=1, size=size):
        actu_chars = np.concatenate((actu_chars, datum['x'].cpu().numpy()))
        actu_codes = np.concatenate((actu_codes, datum['y'].cpu().numpy()))
        actu_ids = np.concatenate((actu_ids, datum['i'].cpu().numpy()))
        
        chars = V(datum['x']) #Get batch data
        pred_codes_raw.append(model(chars).data.cpu()) #model preds
        
        if batch_limit and i >= batch_limit - 1: break # break early if reached batch_limit

    pred_codes_raw = np.concatenate(pred_codes_raw, axis=0)
    pred_codes = np.argmax(pred_codes_raw, axis=2) #Predicted codes are the argmax
    
    return (pred_codes, pred_codes_raw, actu_chars, actu_codes, list(actu_ids))

def getWordPredictions(predicted_codes_raw, actual_chars, actual_codes, char_vocab):
    
    all_space_indices = [[j for j, char in enumerate(chars) if char == char_vocab.index(' ')] for chars in actual_chars]
    predicted_word_codes = []
    predicted_word_codes_raw = []
    actual_word_codes = []
    
    for ci, space_indices in enumerate(all_space_indices):
        last_indice = -1
        word_preds = []
        word_codes = []
        word_preds_raw = []
        for ii in range(len(space_indices)+1):
            try:
                indice = space_indices[ii]
            except:
                #if this is past the last index, go to the index of the first pad
                indice = len([a for a in actual_chars[ci] if a != char_vocab.index('<PAD>')])

            #Averaging for now, but could use product
            word_preds_raw.append(np.absolute(np.average(predicted_codes_raw[ci][last_indice + 1:indice], axis=0)))
            word_codes.append(np.min(actual_codes[ci][last_indice + 1:indice], axis=0))
            word_preds.append(np.argmin(word_preds_raw[-1]))
            last_indice = indice
            if indice == 299: break #if space is the last character, then break the loop
        
        predicted_word_codes.append(word_preds)
        predicted_word_codes_raw.append(word_preds_raw)
        actual_word_codes.append(word_codes)
        
    return predicted_word_codes, predicted_word_codes_raw, actual_word_codes

def totalAccuracy(actuals, predictions):
    if type(actuals) == type(predictions) == list:
        flat_actuals = [a for b in actuals for a in b]; flat_predictions = [a for b in predictions for a in b]
    elif type(actuals) == type(predictions) == np.ndarray:
        flat_actuals = actuals.flatten(); flat_predictions = predictions.flatten()
    combine = list(zip(flat_actuals, flat_predictions))
    a = [t[0] == t[1] for t in combine]
    return sum(a), len(a) # correct, total

def calc_confidence(sentence_raw_word_codes):
    """Provide the raw codes for one sentence"""
    srwc = np.array(sentence_raw_word_codes)
    mins_1 = np.min(srwc, axis=1); mins_1
    mins_1i = np.nanargmin(srwc, axis=1); mins_1i
    for i,m in enumerate(mins_1i):
        srwc[i][m] = None
    mins_2 = np.nanmin(srwc, axis=1); mins_2
    word_conf = list(map(lambda x,y: 1-(x/y),mins_1,mins_2));
    min_word_conf_i = np.argmin(word_conf)
    min_word_conf = min(word_conf); min_word_conf, min_word_conf_i
    return min_word_conf, min_word_conf_i

def calc_confidence2(preds):
    srwc = softmax(preds, axis=2);srwc.shape, np.max(srwc), type(srwc[0][0][0])
    maxs_1 = np.max(srwc.astype('float64'), axis=2); maxs_1.shape, maxs_1, np.min(maxs_1), np.max(maxs_1)
    maxs_1i = np.nanargmax(srwc.astype('float64'), axis=2); maxs_1i.shape, maxs_1i
    for i,m in enumerate(maxs_1i):
        for j, n in enumerate(m):
            srwc[i][j][n] = None
    maxs_2 = np.nanmax(srwc.astype('float64'), axis=2); maxs_2, np.min(maxs_2), np.max(maxs_2)
    word_conf = np.array(list(map(lambda x,y: 1 - (1-x)/(1-y), maxs_1, maxs_2))); word_conf.shape, word_conf
    min_word_conf = np.min(word_conf, 1); min_word_conf.shape, min_word_conf
    return min_word_conf

def getWordFromChars(char_nums, char_vocab):
    chars = [char_vocab[char] for char in char_nums if char_vocab[char] != '<PAD>']; chars
    spaces = [i for i,char in enumerate(chars) if char == ' ']; spaces
    words = []; start_index = - 1
    for i in range(len(spaces)):
        index = spaces[i]
        words.append(''.join(chars[start_index + 1 : index]))
        start_index = index
    words.append(''.join(chars[start_index + 1:]))

    return words

def softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=axis), axis)

def getAllPredictionData(model, dl):
    """return inputs, targets, preds in one numpy array"""
    INPU = 0; TARG = 1; PRED = 2
    maxlen = max([len(a) for a in dl.dataset.x])
    length = len(dl.dataset)
    bs = dl.batch_size
    
    p_data = np.zeros([3, len(dl.dataset),maxlen], dtype=int) #inpus, targs, preds
    p_conf = np.zeros([len(dl.dataset)], dtype=float)

    for i, (inputs, targets) in enumerate(dl):
        preds = np.array(model.predict_array(inputs)[0].swapaxes(0,1), dtype=float)
        p_conf[i*bs:i*bs+bs] = calc_confidence2(preds)
        p_data[INPU][i*bs:i*bs+bs]=[pad_sequence(list(inpu), maxlen, value=1) for inpu in inputs.permute(1,0).cpu()]
        p_data[TARG][i*bs:i*bs+bs]=[pad_sequence(list(targ), maxlen, value=1) for targ in targets.permute(1,0).cpu()]
        p_data[PRED][i*bs:i*bs+bs]=[pad_sequence(list(pred), maxlen, value=1) for pred in preds.argmax(2)]
    
    # Get the percentage of correct sentences and words
    ones = np.sum(p_data[TARG] == 1); ones
    word_errors = np.sum(p_data[TARG] != p_data[PRED]); word_errors
    word_total = np.prod(p_data[TARG].size); word_total
    sent_errors = sum(np.sum(p_data[TARG] != p_data[PRED], axis=1) != 0); sent_errors
    #print('Word accuracy (non-pad): {:.5} %'.format((1 - word_errors / (word_total - ones)) * 100))
    #print('Sentence accuracy: {:.5} %'.format( (1 - sent_errors / length) * 100))
    #print('Total sentences:', length)
    #print('Total sentence errors:', sent_errors)
    
    err_mask = np.sum(p_data[TARG] != p_data[PRED], axis=1) != 0
    err_ids = np.array(list(range(length)))[err_mask]

    return p_data, err_ids, p_conf

def make_connections(etymology_words, etymology_codes, entry_id):
    connections = []
        
    code_info = {}
    if type(etymology_codes) == str: etymology_codes = eval(etymology_codes)
    if type(etymology_words) == str: etymology_words = etymology_words.split(' ')

    ## Get the index
    for i, code in enumerate(etymology_codes):
        code = str(code)
        if code == 'key': code = '[0]'
        if code[0]=='[': # to tell if it is a code
            code_info.setdefault(code, {}).setdefault('indices',[]).append(i)

    ## Get the word
    for code, code_info_dict in code_info.items(): # go through all codes
        word_parts = []
        for word_index in code_info_dict['indices']:
            word_parts.append(etymology_words[word_index])
        code_info_dict['word'] = ' '.join(word_parts)

    ## Infer the language
    for code, code_info_dict in code_info.items(): # go through all codes
        language_parts = []
        for pre_code_index in reversed(range(code_info_dict['indices'][0])):
            if etymology_codes[pre_code_index] == 'lang':
                language_parts.insert(0, etymology_words[pre_code_index])
            else:
                if language_parts: break # break out of this loop
        code_info_dict['lang'] = ' '.join(language_parts)
        # Could be issue if 2 different langs are separated by only a space i.e. "French English"
        # Could test here for bad langauges

    ## make the connections
    for code_string in code_info.keys():
        code = eval(code_string)
        if code_string != '[0]':
            root = code_info[code_string]
            desc = code_info[str(code[:-1])]
            connections.append({
                'root': {i:root[i] for i in root if i!='indices'}, # Add 'word' and 'lang' items
                'desc': {i:desc[i] for i in desc if i!='indices'}, # Add 'word' and 'lang' items
                'source': entry_id}) ## Append the source (table id)

#         print(code_info); print(connections); break
    return connections

##############################################
##############################################
############### MANUAL UPDATING ##############
##############################################
##############################################

def saveCode(dataframe, id_, code, source):
    dataframe.at[id_,'code'] = code
    dataframe.at[id_,'code_source'] = source
    
def save_items(pred_codes, actu_words):
    last_index = None
    for datum in zip(pred_codes, actu_words):
        code = [code_vocab[cv] for cv in datum[0]]; words = getWordFromChars(datum[1])
        pe = ' '.join(words)
        try:
            item = table.query("parsed_etymology == @pe").iloc[0]
        except IndexError as e:
            print('error with', words)
            continue
#             print(code, words)
#             raise e
        i = item.name
        
        if item['code_source'] != 'none': continue
        
        print('id:', i, '; word:', item['word'], 'language:', item['language'], '; last_index:', last_index)
        print('\n'+item['word'])
        print(show_word_analysis(code, None, words))

        print(len(code), code)
#         print(len(parsed_etymology), 'Etymology Length')
#         print(['lang', 'key'] + ['None'] * (len(parsed_etymology) - 4) + [[0 ,0], 'None'])
        next_step = input('[s]ave code, [m]ark as wrong, [c]hange(+code) or []continue? ')
        if next_step == '':
            pass
        elif next_step == 's':
            saveCode(i, code, 'extra_complex_tiramisu_val_break_model_04-27-18_17-05-17')
        elif next_step == 'm':
            pass
            #mark the code as wrong ('##Wrong')
        elif next_step[0] == 'c':
            if len(eval(next_step[1:])) != len(words): raise Exception('lengths dont match')
            clear_output()
            print(show_word_analysis(eval(next_step[1:]), None, words))
            if input('Correct? [y]/n') != 'n':
                saveCode(i, eval(next_step[1:]), 'manual_update')
        clear_output()

        last_index = i
# except KeyboardInterrupt:
#     print('stopped')

def showCodeOrEstimation(row):
    if row['code'] != row['code']: #If code is NaN
        langs = len(row['language'].split(' '))
        keys = len(row['word'].split(' '))
        length = len(row['parsed_etymology'].split(' '))
        return ['lang']*langs + ['key']*keys + ['None']*(length-langs-keys)
    else: #Otherwise give the code
        code = row['code']
        if type(code) == str: code = eval(code)
        return code

def manual_prediction_verifications(dataframe, indices, auto_update_text, manual_update_text='manual_update'):
    try:
        last_index = None
        for index in indices:
            row = dataframe.loc[index]; i = row.name; 
            print('last_index:', last_index)
            print(show_word_analysis(row)) # Show list of pred,actu,sent vertically
            print('Code:', showCodeOrEstimation(row)) # Show the code or ['None']*len
            
            next_step = input('[s]ave code, [m]ark as wrong, [c]hange(+code) or []continue? ')
            if next_step == '' or next_step == 'm':
                pass
            elif next_step == 's':
                if 'pred_code' in row:
                    saveCode(dataframe, i, row['pred_code'], auto_update_text)
                else:
                    raise Exception("No prediction to save! Column 'pred_code' is missing")
            elif next_step[0] == 'c':
                if len(eval(next_step[1:])) != len(row['parsed_etymology'].split(' ')): raise Exception('lengths dont match')
                clear_output()
                row['pred_code'] = eval(next_step[1:])
                print(show_word_analysis(row))
                
                if input('Correct? [y]/n]') != 'n':
                    saveCode(dataframe, i, eval(next_step[1:]), manual_update_text)
            clear_output()
            last_index = i
    except KeyboardInterrupt as k:
        print('stopped')

##############################################
##############################################
############### MASS UPDATING ################
##############################################
##############################################

def add_lang_to_temp_codes(row, languages, max_lang_len=4):
    codes = row['temp_code']
    words = row['parsed_etymology'].split(' ')
    for start in range(len(codes)):
        for length in reversed(range(1, max_lang_len + 1)):
            if start+length>len(codes): continue
            if not all([code == '-' for code in codes[start:start+length]]): continue #continue if not all 'None'

            if ' '.join(words[start:start+length]) in languages.values:
                codes[start:start+length] = ['l'] * length
    return codes

def reduce_temp_code(row, reduce_hyphen=False):
    tc = row['temp_code']
    tc_c = tc + ['***'] 
    if       reduce_hyphen: reduced_temp_code = [code for i,code in enumerate(tc) if code != tc_c[i+1]]
    elif not reduce_hyphen: reduced_temp_code = [code for i,code in enumerate(tc) if code != tc_c[i+1] or code == '-']
    return reduced_temp_code

def make_temp_code(row, bad_words):
    temp_codes = []
    for word in row['parsed_etymology'].split(' '):
        if word in ['from', 'From', '<']: 
            temp_codes.append('f')
        elif word == ',':
            temp_codes.append('c')
        elif word == ':':
            temp_codes.append(':')
        elif word == '+' or word == 'and':
            temp_codes.append('+')
        elif word == '(' or word == '[':
            temp_codes.append('(')
        elif word == ')' or word == ']':
            temp_codes.append(')')
        elif word == ';' or word == '.':
            temp_codes.append('.')
        elif word == 'or' or word == '/':
            temp_codes.append('or/')
        elif word.replace('\'', '') in bad_words:
            temp_codes.append('b')
        else:
            temp_codes.append('-')
    return temp_codes

##############################################
##############################################
############### ERROR ANALYSIS ###############
##############################################
##############################################

class CodeError(IntEnum):
    KEYCODE_NO_PREDECESSOR = 1
    KEYCODE_NO_SIBLING = 2
    KEYCODE_DISCONNECTED = 3
    KEYCODE_AFTER_MEANINGLESS = 4
    KEYCODE_INSIDE_PARENS = 5
    KEYCODE_BAD_LETTER = 6
    LANGUAGE_LOWERCASE = 7
    SIZE_MISMATCH = 8
    
#Updated 6/12
def hasErrors(codes, words, verbose=False):
    """ 
    Parameters:
        codes - codes for each word in the form on 'none', 'lang', '[0, 0]', etc
        words - words for the sentence
    Returns:
        False - if no errors
        error_list -  broken keycode link, nonadjacent keycode instances, lowercase lang, code after see/compare
    """
    error_list = [] 
    code_info = {}
    if codes != codes: return None # return none if there is no code
    if type(codes) == str: codes = eval(codes)
    if type(words) == str: words = words.split(' ')
    # words = row['parsed_etymology'].split(' '); codes = row['code'] # If using a DF row of pandas
    
    def addError(error_data):
        if verbose: print(error_data)
        error_list.append(error_data[0])
    
    # check for length mismatch
    size_diff = len(words) - len(codes)
    if size_diff != 0:
        addError([CodeError.SIZE_MISMATCH, size_diff])
        if size_diff < 0: codes = codes[:len(words)] # shorten code
        if size_diff > 0: codes = codes + ['None']*size_diff # shorten code
        
    # Get the information for each code
    for i, code in enumerate(codes):
        code = str(code)
        if code == 'key': code = '[0]'
        code_info.setdefault(code, {}).setdefault('indices',[]).append(i)

    # See if each code has connections
    for key in code_info.keys():
        if key[0]=='[' and key != '[0]':
            if str(eval(key)[:-1]) not in code_info.keys():
                addError([CodeError.KEYCODE_NO_PREDECESSOR, key])
            if str(eval(key)[:-1] + [max(eval(key)[-1] - 1 , 0)]) not in code_info.keys():
                addError([CodeError.KEYCODE_NO_SIBLING, key])
            
            # INSIDE PARENTHESES
            paren_count = 0
            for w in ''.join(words[max(code_info[key]['indices']):]): #check all letters after keycode for parens
                if w == '(': paren_count+=1
                elif w == ')': paren_count-=1
            if paren_count != 0:
                addError([CodeError.KEYCODE_INSIDE_PARENS, key])

            # BAD LETTERS FOR KEYCODE "(", "+", ETC
            # OTHER BAD WORDS (TO ADD): ?, unknown, \n\n
            for w_index in code_info[key]['indices']:
                if words[w_index] in '+=-_()[]':
                    addError([CodeError.KEYCODE_BAD_LETTER])
                
    # See if the codes appear multiple times
        if key[0]=='[':
            indices = code_info[key]['indices']
            if max(indices) - min(indices) + 1 != len(indices):
                addError([CodeError.KEYCODE_DISCONNECTED, [key, indices]])

    # Check for languages without capitals
    code_info.setdefault('lang', {'indices': []})
    lc_langs = [words[i] for i in code_info['lang']['indices'] if words[i].lower() == words[i]]
    if lc_langs:
        addError([CodeError.LANGUAGE_LOWERCASE, lc_langs])

    # code after See or compare
    meaningless_words = ['see', 'compare', 'cognate']
    meaningless_index = min([i for i, word in enumerate(words) if word.lower() in meaningless_words]+[sys.maxsize])
    all_keycode_indices = [item for k,v in code_info.items() for item in v['indices'] if k[0]=='[' and item > meaningless_index] 
    
    #flat list of all indices past meaningless index
    if all_keycode_indices:
        addError([CodeError.KEYCODE_AFTER_MEANINGLESS, [meaningless_words, ':', [codes[i] for i in all_keycode_indices]]])

    # BAD LANGUAGES
    
    return error_list # if there are no errors then this is empty
