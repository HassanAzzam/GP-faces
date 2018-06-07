import gensim.models.keyedvectors as word2vec
import argparse
import pickle
import time
import os
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from collections import OrderedDict

start_time = time.time()

print 'Loading FastText model...'
#model = word2vec.KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec', binary=False, limit=999993)
model = word2vec.KeyedVectors.load_word2vec_format('crawl-300d-2M.vec', binary=False, limit=1999999)
#model = {}
load_time = time.time() - start_time

print 'loaded model in ' + str(load_time) + ' seconds'

def preprocess(text):
        """
        Preprocess text for encoder
        """
        X = []
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        for t in text:
                sents = sent_detector.tokenize(t)
                result = ''
                for s in sents:
                        #s = s.replace('is\'t', 'isn\'t')
                        s = s.replace('hairbrown ', '')
                        s = s.replace('blound', 'blond')
                        #s = s.replace('hishairline', 'his hairline')
                        #s = s.replace('herhairline', 'her hairline')
                        s = s.replace('5_o_clock shadow', 'stubble')
                        #s = s.replace('5_o_clock_shadow', 'stubble')
                        #s = s.replace('five o\'clock shadow', 'stubble')
                        #s = s.replace('5 o\'clock shadow', 'stubble')
                        #s = s.replace('n\'t', 'not')
                        tokens = word_tokenize(s)
                        result += ' ' + ' '.join(tokens)
                result = result.replace('n\'t', 'not')
                X.append(result.lower().split())
        return X

def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('--captions_folder', type=str, default='Data/faces/captions/',
                                                help='captions folder')
        parser.add_argument('--data_dir', type=str, default='Data',
                                                help='Data Directory')

        global model
        args = parser.parse_args()

        # start_time = time.time()

        # print 'Loading FastText model...'
        # #model = word2vec.KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec', binary=False, limit=999993)
        # #model = word2vec.KeyedVectors.load_word2vec_format('crawl-300d-2M.vec', binary=False, limit=1999999)
        # load_time = time.time() - start_time

        # print 'loaded model in ' + str(load_time) + ' seconds'

        captions_files = [fn for fn in os.listdir(args.captions_folder) if 'txt' in fn]

        print '%s caption files... ' % str(len(captions_files))

        caption_vectors = {}
        for i, fn in enumerate(sorted(captions_files)):
                        with open(os.path.join(args.captions_folder, fn), 'r') as f:
                                sentences = preprocess(f.read().split('\n'))
                                # print fn
                                # print sentences
                                # if i < 1:
                                #         continue
                                # exit()
                                embedded_sentences = []
                                print 'Embedding {0}/{1}... '.format(i+1, len(captions_files))
                                for sent in sentences:
                                        sent = list(map(lambda x: model[x], sent))
                                        embedded_sentences.append(sent)
                                caption_vectors[fn[:-4]] = np.array(embedded_sentences)
                                
        #print caption_vectors

        # a = { '000001': [[1, 2, 3], [4, 5]], '000002': [[3]]}
        with open(os.path.join(args.data_dir, 'caption_vectors.pickle'), 'wb') as handle:
                pickle.dump(caption_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # with open('filename.pickle', 'rb') as handle:
        #         b = pickle.load(handle)

        # if os.path.isfile(os.path.join(args.data_dir, 'caption_vectors.hdf5')):
        #         os.remove(os.path.join(args.data_dir, 'caption_vectors.hdf5'))
        # h = h5py.File(os.path.join(args.data_dir, 'caption_vectors.hdf5'), mode='a')
        # h.create_dataset('vectors', data=caption_vectors)		
#print(model['eyebrows'])

def embed_sentence(sent, max_length=620):
        
        sent = preprocess([sent])
        sent[0] += ['.'] * (max_length - len(sent))
        print sent[0]
        sent = list(map(lambda x: model[x], sent[0]))

        return sent


if __name__ == '__main__':
        main()