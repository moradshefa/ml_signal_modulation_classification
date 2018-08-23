
# coding: utf-8

# This notebook does t-SNE on the army signal classification data set using our model. 
# 
# It Takes 100 points from each modulation from a training set and a test set. It only uses data from the modulations specified by all_mods and mod_group. The current mod_group is set to 6 which was a group that, using t-SNE, we noticed our model was having a tough time to distinguish.
# 
# Since we dont have labels for the testset we first filter using our base model to get the modulations we are interested in. We then run another model (embed_model) on that filtered data. Then use t-SNE in hopes of evaluating how well embed_model does in distinguising the modulations.
# 
# Most of the code is to preprocess the data, filter and label the testset in order to do t-SNE
# which was a valuable tool in determining good models without having labelled test data or being able to
# submit frequently.
# 
# Note, this is not using tsne_plot from tsne_utils.py but it is basically doing the same thing. This was for quick evaluation using many different model so I did not want to have to redo all the work when deciding I wanted to hide the legend for example. For reference on how to use tnse_plot there is tsne_train_only.ipynb and mnist_tsne.ipynb

# In[1]:

from data_loader import *
from keras.models import Model, load_model
from keras import backend as K
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from bokeh.plotting import figure, output_notebook, show, ColumnDataSource
from bokeh.models import HoverTool


# In[2]:

CLASSES_24 = ['16PSK', '2FSK_5KHz', '2FSK_75KHz', '8PSK', 'AM_DSB', 'AM_SSB', 'APSK16_c34',
 'APSK32_c34', 'BPSK', 'CPFSK_5KHz', 'CPFSK_75KHz', 'FM_NB', 'FM_WB',
 'GFSK_5KHz', 'GFSK_75KHz', 'GMSK', 'MSK', 'NOISE', 'OQPSK', 'PI4QPSK', 'QAM16',
 'QAM32', 'QAM64', 'QPSK']

BOOKEH_COLORS = {
    '16PSK': 'aqua', 
    '16PSK_TS1': 'aqua', 
    '2FSK_5KHz': 'aquamarine', 
    '2FSK_5KHz_TS1': 'aquamarine', 
    '2FSK_75KHz': 'bisque', 
    '2FSK_75KHz_TS1': 'bisque', 
    '8PSK': 'black', 
    '8PSK_TS1': 'black', 
    'AM_DSB': 'blue', 
    'AM_DSB_TS1': 'blue', 
    'AM_SSB':'blueviolet', 
    'AM_SSB_TS1':'blueviolet', 
    'APSK16_c34': 'brown',
    'APSK16_c34_TS1': 'brown',
    'APSK32_c34': 'burlywood', 
    'APSK32_c34_TS1': 'burlywood', 
    'BPSK': 'cadetblue', 
    'BPSK_TS1': 'cadetblue', 
    'CPFSK_5KHz': 'chartreuse', 
    'CPFSK_5KHz_TS1': 'chartreuse', 
    'CPFSK_75KHz': 'chocolate', 
    'CPFSK_75KHz_TS1': 'chocolate', 
    'FM_NB': 'cornflowerblue', 
    'FM_NB_TS1': 'cornflowerblue', 
    'FM_WB': 'crimson',
    'FM_WB_TS1': 'crimson',
    'GFSK_5KHz': 'darkcyan', 
    'GFSK_5KHz_TS1': 'darkcyan', 
    'GFSK_75KHz': 'darkgoldenrod', 
    'GFSK_75KHz_TS1': 'darkgoldenrod', 
    'GMSK': 'darkgray', 
    'GMSK_TS1': 'darkgray', 
    'MSK': 'darkgreen', 
    'MSK_TS1': 'darkgreen', 
    'NOISE': 'darkorange', 
    'NOISE_TS1': 'darkorange', 
    'OQPSK': 'deeppink', 
    'OQPSK_TS1': 'deeppink', 
    'PI4QPSK': 'fuchsia', 
    'PI4QPSK_TS1': 'fuchsia', 
    'QAM16': 'gold',
    'QAM16_TS1': 'gold',
    'QAM32': 'lightblue', 
    'QAM32_TS1': 'lightblue', 
    'QAM64': 'magenta', 
    'QAM64_TS1': 'magenta', 
    'QPSK': 'plum',
    'QPSK_TS1': 'plum'
}


BOOKEH_SHAPES = {
    '16PSK':1,
    '16PSK_TS1':2, 
    '2FSK_5KHz':1,
    '2FSK_5KHz_TS1':2, 
    '2FSK_75KHz':1,
    '2FSK_75KHz_TS1':2, 
    '8PSK':1,
    '8PSK_TS1':2, 
    'AM_DSB':1,
    'AM_DSB_TS1':2, 
    'AM_SSB':1,
    'AM_SSB_TS1':2, 
    'APSK16_c34':1,
    'APSK16_c34_TS1':2,
    'APSK32_c34':1,
    'APSK32_c34_TS1':2, 
    'BPSK':1,
    'BPSK_TS1':2, 
    'CPFSK_5KHz':1,
    'CPFSK_5KHz_TS1':2, 
    'CPFSK_75KHz':1,
    'CPFSK_75KHz_TS1':2, 
    'FM_NB':1,
    'FM_NB_TS1':2, 
    'FM_WB':1,
    'FM_WB_TS1':2,
    'GFSK_5KHz':1,
    'GFSK_5KHz_TS1':2, 
    'GFSK_75KHz':1,
    'GFSK_75KHz_TS1':2, 
    'GMSK':1,
    'GMSK_TS1':2, 
    'MSK':1,
    'MSK_TS1':2, 
    'NOISE':1,
    'NOISE_TS1':2, 
    'OQPSK':1,
    'OQPSK_TS1':2, 
    'PI4QPSK':1,
    'PI4QPSK_TS1':2, 
    'QAM16':1,
    'QAM16_TS1':2,
    'QAM32':1,
    'QAM32_TS1':2, 
    'QAM64':1,
    'QAM64_TS1':2, 
    'QPSK':1,
    'QPSK_TS1':2,
}


# In[3]:

def tsne_model(model, data, pca_dim=50, tsne_dim=2, preds = False, layer_index = -2):
    """
    Does tsne reduction
    Parameters:
    model (keras model)
    data (np array): input data for the model
    layer_name: name of the output layer to do tsne on. 'None' will use 2nd to last layer, before final dense layer
 
    pca_dim (int): first dimensions are reduced using PCA to this number of dimensions
    tsne_dim (int): final desired dimension after doing tsne reduction
    
    Returns:
    np.array: Shape (data.shape[0], tsne_reduc); sample points in reduced space
    """
    
    layer_name = model.layers[layer_index].name
    
    intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
    
    intermediate_output = intermediate_layer_model.predict(data)

    pca = PCA(n_components= pca_dim, random_state = 214853)    
    output_pca_reduced = pca.fit_transform(intermediate_output)
    
    tsne = TSNE(n_components=tsne_dim, random_state=214853)
    intermediates_tsne = tsne.fit_transform(output_pca_reduced)
    
    if preds:
        return intermediates_tsne, model.predict(data)
    return intermediates_tsne

def load_training_data(data_file,num_samples=100, mods = None, spectrum=False):
    testdata = LoadModRecData(data_file, 1., 0., 0., load_snrs=[10], num_samples_per_key=num_samples, load_mods = mods,spectrum=spectrum)
    train_data = testdata.signalData
    train_labels = testdata.signalLabels[:,0]
    return train_data, train_labels


def open_test_file(test_file, snr_model, filter_snr = True):
    # opens a testfile and if filter_snr is set to true then it will filter 
    # out only samples that have been predicted to have snr 10dB
    f = open(test_file, 'rb')
    testdata = pickle.load(f, encoding='latin1')
    testdata = np.asarray([testdata[i+1] for i in range(len(testdata.keys()))])

    if filter_snr:
        snr_probs = snr_model.predict(testdata)
        snr_preds = np.asarray([np.argmax(snr_prob) for snr_prob in snr_probs])
        testdata = testdata[np.where(snr_preds == 5)]
    
    return testdata

      
def get_mods_test(model, data, mods, classes):
    # takes in test set returns those data samples that classify in the group of mods by model

    preds = model.predict(data)
    preds = np.asarray([np.argmax(pred) for pred in preds])
    labels = np.asarray([classes[pred] for pred in preds])

    idx = []
    for i,labl in enumerate(preds):
        if labl in mods:
            idx.append(i)
    idx = np.asarray(idx)
    
    return data[idx]


def separate_labels(tsne_output, labels, preds):
    # splits tsne output, labels, and preds into subarrays that have the same labels
    unique_labels = np.unique(labels)
    
    tsne_subar = [0] * unique_labels.shape[0]
    label_subar = [0] * unique_labels.shape[0]
    preds_subar = [0] * unique_labels.shape[0]
    indices = [0] * unique_labels.shape[0]
    for i,label in enumerate(unique_labels):
        idx = np.where(labels==label)
        
        tsne_subar[i] = tsne_output[idx]
        label_subar[i] = labels[idx]
        preds_subar[i] = preds[idx]
        indices[i] = idx[0]
        
    return tsne_subar, label_subar, indices, preds_subar


# In[4]:

snr_model = load_model("../../snr2.h5")
# model = load_model('../../mod_group0_val_loss5754_copy.h5')
model = load_model('../../mod_group0_val_loss0546.h5')


# In[5]:

model0_path = '../../mod_group0_val_loss0546.h5'
snr_model_path = '../../snr2.h5'

train_file_path = "/datax/yzhang/training_data/training_data_chunk_14.pkl"
train_file_path = "/datax/yzhang/army_challenge/training_data/training_data_chunk_14.pkl"

test_file_path = "../../Test_Set_1_Army_Signal_Challenge.pkl"

num_samples_from_train = 100
num_samples_from_test = 100

mod_group=6


# In[6]:

model = load_model(model0_path)
snr_model = load_model(snr_model_path)


# In[7]:

all_mods = [np.arange(24), np.array([1,9,10,11,12,13]), 
            np.array([4,5]), np.array([1,9]), np.array([6,7,20,21,22]), np.array([0,3]), np.array([0,3,6,7,20,21,22])]

mods = all_mods[mod_group]


train_data, train_labels = load_training_data(train_file_path,num_samples=num_samples_from_train,mods = [CLASSES_24[i] for i in mods], spectrum=False)


# In[8]:

testdata1_filtered_snr = open_test_file(test_file_path, snr_model, filter_snr = True)


# In[9]:

test_data_ = get_mods_test(model, testdata1_filtered_snr, mods, CLASSES_24)


# In[10]:

#########################
# change with new model #
#########################
embed_model_path = '../../tmp_gp6_da/model_fancydis_1.h5'
embed_model = load_model(embed_model_path)
layer_index = None


# In[11]:

# pick layer index
embed_model.summary()


# In[12]:

#########################
# change with new model #
#########################
layer_index = -6


# In[13]:

# When trying a new model to save time do this:
    # load the model
    # pick the layer_index
    # start running from here

test_data = np.copy(test_data_)
preds = embed_model.predict(test_data)
preds = np.asarray([np.argmax(pred) for pred in preds])
test_labels = np.asarray([CLASSES_24[mods[pred]]+"_TS1" for pred in preds])


# In[14]:

inter_data, inter_labels = [], []
for labl in np.unique(test_labels):
    idx = np.where(test_labels == labl)
    inter_data.append(test_data[idx][:num_samples_from_test])
    inter_labels.append(test_labels[idx][:num_samples_from_test])

test_data = np.concatenate(inter_data)
test_labels = np.concatenate(inter_labels)


# In[15]:

data = np.concatenate((train_data, test_data))
labels = np.concatenate((train_labels, test_labels))

print(data.shape, labels.shape)


# In[16]:

tsne_output, preds = tsne_model(model=embed_model,data=data, preds=True, layer_index=-6)

# round to 3 decimals
preds = np.around(preds, decimals=3)

# turn each prediction into a list of strings
preds = np.asarray([[CLASSES_24[mods[i]] +": " + str(pred[i]) for i in range(len(pred))] for pred in preds])


# In[17]:

tsne_sub, labels_sub, indices, preds_sub = separate_labels(tsne_output, labels, preds)


# In[18]:

legend = False # will hide legend if it gets annoying
# legend = 'label'


# In[19]:

tooltips = [("p", "(@x, @y)"),("label", "@label"),("index", "@index"),("pred", "@pred")]

hover_tsne = HoverTool(tooltips = tooltips) 
tools_tsne = [hover_tsne, 'pan', 'wheel_zoom', 'reset','box_zoom','save']
p = figure(plot_width=700, plot_height=700, tools=tools_tsne)
output_notebook()

for i in range(len(tsne_sub)):
    tsne_ = tsne_sub[i]
    labels_ = labels_sub[i]
    indices_ = indices[i]
    preds_ = preds_sub[i]
    labl = labels_[0]

    source_train = ColumnDataSource(
        data=dict(
            x = tsne_[:,0],
            y = tsne_[:,1],
            index = indices_,
            label = labels_,
            pred = preds_
        )
    )
    

    shape = BOOKEH_SHAPES[labl]
    if shape == 1:
        p.circle('x', 'y', size=7, fill_color=BOOKEH_COLORS[labl], 
                 alpha=0.9, line_width=0, source=source_train, name="test", legend=legend)
    elif shape == 2:
        p.diamond('x', 'y', size=7, fill_color=BOOKEH_COLORS[labl], 
                 alpha=0.9, line_width=0, source=source_train, name="test", legend=legend)
    elif shape == 3:
        p.cross('x', 'y', size=7, fill_color=BOOKEH_COLORS[labl], 
                 alpha=0.9, line_width=0, source=source_train, name="test", legend=legend)

p.legend.click_policy="hide"

show(p)


# In[ ]:



