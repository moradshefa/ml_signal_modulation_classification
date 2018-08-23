from keras import backend as K
from keras.models import Model
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from bokeh.plotting import figure, output_notebook, show, ColumnDataSource
from bokeh.models import HoverTool

import numpy as np

def tsne_model(model, data, pca_dim=50, tsne_dim=2, preds = False, layer_index = -2, random_state = None):
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

    pca = PCA(n_components= pca_dim, random_state = random_state)    
    output_pca_reduced = pca.fit_transform(intermediate_output)
    
    tsne = TSNE(n_components=tsne_dim, random_state=random_state)
    intermediates_tsne = tsne.fit_transform(output_pca_reduced)
    
    if preds:
        return intermediates_tsne, model.predict(data)
    return intermediates_tsne


def separate_labels(tsne_output, labels, preds):
    # splits tsne output, labels, and preds into subarrays that have the same labels (for plotting by label)
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



def tsne_plot(model, data, labels, classes, colors, shapes,layer_index, legend = True, show_preds = True, pca_dim=50, tsne_dim=2):
    """
    Does tsne reduction and plotting
    Parameters:
    model (keras model)
    data (np array): input data for the model
    layer_index: index of the output layer to do tsne on. 
    pca_dim (int): first dimensions are reduced using PCA to this number of dimensions
    tsne_dim (int): final desired dimension after doing tsne reduction
    show_preds (boolean): show prediction probability for each class when hovering over plot
    legend (boolean): show legend
    colors (dict label (String)-> Color (String)): Bookeh colors to use for the plot,  https://bokeh.pydata.org/en/latest/docs/reference/colors.html for referece of choices
    shapes (dict label (String)-> shape (int)): Specify which shape is desired 1 for circle,2 for diamond, 3 for cross (easy to modify to your own desired shapes)
    classes (string array): if show_preds is true specify what prediction belongs to what label, e.g. if the i-th one hot vector corresponds to label 'dog' then classes[i] should be 'dog'
    
    """
    if legend:
        legend = 'label'

    if show_preds:
        tsne_output, preds = tsne_model(model=model,data=data, preds=show_preds, layer_index=layer_index, pca_dim=pca_dim, tsne_dim=tsne_dim)
        # round to 3 decimals
        preds = np.around(preds, decimals=3)

        # turn each prediction into a list of strings
        preds = np.asarray([[classes[i] +": " + str(pred[i]) for i in range(len(pred))] for pred in preds])
    else:
        tsne_output = tsne_model(model=model,data=data, preds=show_preds, layer_index=layer_index, pca_dim=pca_dim, tsne_dim=tsne_dim)
        preds = np.zeros(labels.shape)

    tsne_sub, labels_sub, indices, preds_sub = separate_labels(tsne_output, labels, preds)
    
    
    tooltips = [("p", "(@x, @y)"),("label", "@label"),("index", "@index")]
    
    if show_preds:
        tooltips.append(("preds", "@preds"))
    
    hover_tsne = HoverTool(tooltips = tooltips) 
    tools_tsne = [hover_tsne, 'pan', 'wheel_zoom', 'reset','box_zoom','save']
    p = figure(plot_width=700, plot_height=700, tools=tools_tsne)
    output_notebook()

    for i in range(len(tsne_sub)):
        tsne_ = tsne_sub[i]
        labels_ = labels_sub[i]
        indices_ = indices[i]
        
        if show_preds:
            preds_ = preds_sub[i]
        else:
            preds_ = np.zeros(indices_.shape)

        
        labl = labels_[0]

        source_train = ColumnDataSource(
            data=dict(
                x = tsne_[:,0],
                y = tsne_[:,1],
                index = indices_,
                label = labels_,
                preds = preds_
            )
        )

        shape = shapes[labl]
        if shape == 1:
            p.circle('x', 'y', size=7, fill_color=colors[labl], 
                     alpha=0.9, line_width=0, source=source_train, name="test", legend=legend)
        elif shape == 2:
            p.diamond('x', 'y', size=7, fill_color=colors[labl], 
                     alpha=0.9, line_width=0, source=source_train, name="test", legend=legend)
        elif shape == 3:
            p.cross('x', 'y', size=7, fill_color=colors[labl], 
                     alpha=0.9, line_width=0, source=source_train, name="test", legend=legend)

    p.legend.click_policy="hide"

    show(p)
    
