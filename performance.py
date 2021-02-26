import numpy as np
import scipy
from scipy import ndimage
from scipy.stats import sem
import matplotlib.pyplot as plt

def tri_img_vs_word(word_epoch = 50, img_indices = range(1000), wrd_indices = range(1000,2000), show=1):
    plt.subplots(figsize=(12,5))
    plt.subplots_adjust(left = 0.06, bottom = 0.11, right = 0.94, top = 0.88, wspace = 0.25, hspace = 0.20)
    
    # illit
    plt.subplot(1,3,1)
    cat_scores=np.load('../save/networks/cat_scores_illit_z_full_nomir.npy')
    l,c = np.shape(cat_scores)
    smoothed_scores = ndimage.gaussian_filter1d(cat_scores[:,img_indices], 1, 0)
    x = range(l)
    y = np.mean(smoothed_scores, axis=1)
    error = sem(smoothed_scores, axis=1)
    print ('error', error)
    plt.plot(x, y, color='#089FFF', linewidth=2, label='image mean')
    plt.fill_between(x, y-error, y+error, alpha=0.3, edgecolor='#089FFF', facecolor='#089FFF')
    label2, labelv, title = 'word mean', 'words in', 'Illiterate network'
    plt.xticks(np.linspace(0, 80, 9), [str(int(i)) for i in np.linspace(0, 80, 9)])
    plt.ylabel('Validation Accuracy')
    plt.xlabel('Training epochs')
    plt.title(title, size=13)
    plt.ylim(0,100)
    plt.annotate('A', xy=(0, 0), xytext=(-5, 102), size=20, weight='extra bold')

    # lit_no_bias
    plt.subplot(1,3,2)
    cat_scores=np.load('../save/networks/cat_scores_lit_no_bias_z_full_nomir.npy')
    l,c = np.shape(cat_scores)
    smoothed_scores = ndimage.gaussian_filter1d(cat_scores[:,img_indices], 1, 0)
    smoothed_scores_wrd = ndimage.gaussian_filter1d(cat_scores[word_epoch:,wrd_indices], 1, 0)
    label2, labelv, title = 'word mean', 'words in', 'Unbiased literate network'
    plt.xticks(np.linspace(0, 80, 9), [str(int(i)) for i in np.linspace(0, 80, 9)])
    x = range(l)
    y = np.mean(smoothed_scores, axis=1)
    error = sem(smoothed_scores, axis=1)
    print ('error', error)
    plt.plot(x, y, color='#089FFF', linewidth=2, label='image mean')
    plt.fill_between(x, y-error, y+error, alpha=0.3, edgecolor='#089FFF', facecolor='#089FFF')

    x = range(word_epoch,l)
    y = np.mean(smoothed_scores_wrd, axis=1)
    error = sem(smoothed_scores_wrd, axis=1)
    print ('error', error)
    plt.plot(x, y, color='#1B2ACC', linewidth=2, label=label2)
    plt.fill_between(x, y-error, y+error, alpha=0.3, edgecolor='#1B2ACC', facecolor='#1B2ACC')
    print (scipy.stats.sem(smoothed_scores, axis=1))
    
    plt.axvline(x=word_epoch, ls = '--', linewidth=2, c='red',label=labelv)
    plt.xlabel('Training epochs')
    plt.title(title, size=13)
    plt.ylim(0,100)
    plt.annotate('B', xy=(0, 0), xytext=(-5, 102), size=20, weight='extra bold')
    

    # lit bias
    plt.subplot(1,3,3)
    cat_scores=np.load('../save/networks/cat_scores_lit_bias_z_full_nomir.npy')
    l,c = np.shape(cat_scores)
    smoothed_scores = ndimage.gaussian_filter1d(cat_scores[:,img_indices], 1, 0)
    smoothed_scores_wrd = ndimage.gaussian_filter1d(cat_scores[word_epoch:,wrd_indices], 1, 0)
    label2, labelv, title = 'word mean', 'words in', 'Biased literate network'
    plt.xticks(np.linspace(0, 80, 9), [str(int(i)) for i in np.linspace(0, 80, 9)])
    x = range(l)
    y = np.mean(smoothed_scores, axis=1)
    error = sem(smoothed_scores, axis=1)
    print ('error', error)
    plt.plot(x, y, color='#089FFF', linewidth=2, label='image mean')
    plt.fill_between(x, y-error, y+error, alpha=0.3, edgecolor='#089FFF', facecolor='#089FFF')
        
    x = range(word_epoch,l)
    y = np.mean(smoothed_scores_wrd, axis=1)
    error = sem(smoothed_scores_wrd, axis=1)
    print ('error', error)
    plt.plot(x, y, color='#1B2ACC', linewidth=2, label=label2)
    plt.fill_between(x, y-error, y+error, alpha=0.3, edgecolor='#1B2ACC', facecolor='#1B2ACC')
    print (scipy.stats.sem(smoothed_scores, axis=1))
    
    plt.axvline(x=word_epoch, ls = '--', linewidth=2, c='red',label=labelv)
    plt.legend(loc='upper left', frameon=False, ncol=1, prop={'size': 10})
    plt.xlabel('Training epochs')
    plt.title(title, size=13)
    plt.ylim(0,100)
    plt.annotate('C', xy=(0, 0), xytext=(-5, 102), size=20, weight='extra bold')

    plt.savefig('tri_mean_test_accuracy.png', dpi=300)

    if show:
        plt.show()
    plt.close()
    return 'done'
