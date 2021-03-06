import tensorflow as tf
import numpy as np
import model
import argparse
import pickle
from os.path import join
import h5py
from Utils import image_processing
import scipy.misc
import random
import json
import os
import shutil
import word2vec


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--z_dim', type=int, default=100,
					   help='Noise dimension')

	parser.add_argument('--t_dim', type=int, default=300,
					   help='Text feature dimension')

	parser.add_argument('--batch_size', type=int, default=64,
					   help='Batch Size')

	parser.add_argument('--image_size', type=int, default=64,
					   help='Image Size a, a x a')

	parser.add_argument('--gf_dim', type=int, default=64,
					   help='Number of conv in the first layer gen.')

	parser.add_argument('--df_dim', type=int, default=64,
					   help='Number of conv in the first layer discr.')

	parser.add_argument('--gfc_dim', type=int, default=1024,
					   help='Dimension of gen untis for for fully connected layer 1024')

	parser.add_argument('--sentence_length', type=int, default=620,
					   help='Caption Sentence Length')

	parser.add_argument('--word_embedding_vector_length', type=int, default=300,
					   help='Word Embedding Vector Length')

	parser.add_argument('--caption_vector_length', type=int, default=300,
					   help='Caption Vector Length')

	parser.add_argument('--data_dir', type=str, default="Data",
					   help='Data Directory')

	parser.add_argument('--learning_rate', type=float, default=0.0002,
					   help='Learning Rate')

	parser.add_argument('--beta1', type=float, default=0.5,
					   help='Momentum for Adam Update')

	parser.add_argument('--epochs', type=int, default=100,
					   help='Max number of epochs')

	parser.add_argument('--save_every', type=int, default=30,
					   help='Save Model/Samples every x iterations over batches')

	parser.add_argument('--resume_model', type=str, default=None,
                       help='Pre-Trained Model Path, to resume from')

	parser.add_argument('--data_set', type=str, default="faces",
                       help='Dat set: MS-COCO, flowers')

	args = parser.parse_args()
	model_options = {
		'z_dim' : args.z_dim,
		't_dim' : args.t_dim,
		'batch_size' : args.batch_size,
		'image_size' : args.image_size,
		'gf_dim' : args.gf_dim,
		'df_dim' : args.df_dim,
		'gfc_dim' : args.gfc_dim,
		'caption_vector_length' : args.caption_vector_length,
		'word_embedding_vector_length' : args.word_embedding_vector_length,
		'sentence_length' : args.sentence_length,
	}
	
	
	gan = model.GAN(model_options)
	input_tensors, variables, loss, outputs, checks = gan.build_model()
	with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
		d_optim = tf.train.AdamOptimizer(args.learning_rate, beta1 = args.beta1).minimize(loss['d_loss'], var_list=variables['d_vars'])
		g_optim = tf.train.AdamOptimizer(args.learning_rate, beta1 = args.beta1).minimize(loss['g_loss'], var_list=variables['g_vars'])
	
	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()
	
	saver = tf.train.Saver()
	if args.resume_model:
		saver.restore(sess, args.resume_model)
	
	loaded_data = load_training_data(args.data_dir, args.data_set)
	
	for i in range(args.epochs):
		batch_no = 0
		list_losses_d=[]
		list_losses_g = []
		list_batches=[]
		while batch_no*args.batch_size < loaded_data['data_length']:
			real_images, wrong_images, caption_vectors, z_noise, image_files = get_training_batch(batch_no, args.batch_size, 
				args.image_size, args.z_dim, args.word_embedding_vector_length, 'train', args.data_dir, args.data_set, loaded_data)
			
			# DISCR UPDATE
			check_ts = [ checks['d_loss1'] , checks['d_loss2'], checks['d_loss3']]
			_, d_loss, gen, d1, d2, d3 = sess.run([d_optim, loss['d_loss'], outputs['generator']] + check_ts,
				feed_dict = {
					input_tensors['t_real_image'] : real_images,
					input_tensors['t_wrong_image'] : wrong_images,
					input_tensors['t_real_caption'] : caption_vectors,
					input_tensors['t_z'] : z_noise,
				})
			
			print "d1", d1
			print "d2", d2
			print "d3", d3
			print "D", d_loss
			
			# GEN UPDATE
			_, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']],
				feed_dict = {
					input_tensors['t_real_image'] : real_images,
					input_tensors['t_wrong_image'] : wrong_images,
					input_tensors['t_real_caption'] : caption_vectors,
					input_tensors['t_z'] : z_noise,
				})

			# GEN UPDATE TWICE, to make sure d_loss does not go to 0
			_, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']],
				feed_dict = {
					input_tensors['t_real_image'] : real_images,
					input_tensors['t_wrong_image'] : wrong_images,
					input_tensors['t_real_caption'] : caption_vectors,
					input_tensors['t_z'] : z_noise,
				})
			list_losses_d.append(d_loss)
			list_losses_g.append(g_loss)
			list_batches.append(batch_no)
			print "LOSSES", d_loss, g_loss, batch_no, i, len(loaded_data['image_list'])/ args.batch_size
			batch_no += 1
			if (batch_no % args.save_every) == 0:
				print "Saving Images, Model"
				save_for_vis(args.data_dir, real_images, gen, image_files)
				save_path = saver.save(sess, "Data/Models/latest_model_{}_temp.ckpt".format(args.data_set))
		if i%5 == 0:
			save_path = saver.save(sess, "Data/Models/model_after_{}_epoch_{}.ckpt".format(args.data_set, i))
		with open("Data/plots/losses_discriminator_epoch_{}.txt".format(i), 'w') as f:
			for s in list_losses_d:
				f.write(str(s) + '\n')
		with open("Data/plots/losses_generator_epoch_{}.txt".format(i), 'w') as f:
			for s in list_losses_g:
				f.write(str(s) + '\n')
				
def load_training_data(data_dir, data_set):
	# h = h5py.File(join(data_dir, 'celebA_captions.hdf5'))
	# flower_captions = {}
	# for ds in h.iteritems():
	# 	flower_captions[ds[0]] = np.array(ds[1])
	image_list = [fn[:-4] for fn in os.listdir(os.path.join(data_dir, 'faces/jpg')) if 'jpg' in fn]
	image_list.sort()

	img_75 = int(len(image_list))
	training_image_list = image_list[0:img_75]
	random.shuffle(training_image_list)
	
	return {
		'image_list' : training_image_list,
		#'captions' : flower_captions,
		'data_length' : len(training_image_list)
	}
	

def save_for_vis(data_dir, real_images, generated_images, image_files):
	
	shutil.rmtree( join(data_dir, 'samples') )
	os.makedirs( join(data_dir, 'samples') )

	for i in range(0, real_images.shape[0]):
		real_image_255 = np.zeros( (64,64,3), dtype=np.uint8)
		real_images_255 = (real_images[i,:,:,:])
		scipy.misc.imsave( join(data_dir, 'samples/{}_{}.jpg'.format(i, image_files[i].split('/')[-1] )) , real_images_255)

		fake_image_255 = np.zeros( (64,64,3), dtype=np.uint8)
		fake_images_255 = (generated_images[i,:,:,:])
		scipy.misc.imsave(join(data_dir, 'samples/fake_image_{}.jpg'.format(i)), fake_images_255)


def get_training_batch(batch_no, batch_size, image_size, z_dim, 
	caption_vector_length, split, data_dir, data_set, loaded_data = None):
	real_images = np.zeros((batch_size, 64, 64, 3))
	wrong_images = np.zeros((batch_size, 64, 64, 3))
	captions = np.zeros((batch_size, caption_vector_length))

	cnt = 0
	image_files = []
	for i in range(batch_no * batch_size, batch_no * batch_size + batch_size):
		idx = i % len(loaded_data['image_list'])
		image_file =  join(data_dir, 'faces/jpg/'+loaded_data['image_list'][idx]+'.jpg')
		image_array = image_processing.load_image_array(image_file, image_size)
		real_images[cnt,:,:,:] = image_array
		
		# Improve this selection of wrong image
		wrong_image_id = random.randint(0,len(loaded_data['image_list'])-1)
		wrong_image_file =  join(data_dir, 'faces/jpg/'+loaded_data['image_list'][wrong_image_id]+'.jpg')
		wrong_image_array = image_processing.load_image_array(wrong_image_file, image_size)
		wrong_images[cnt, :,:,:] = wrong_image_array

		random_caption = random.randint(0,4)
		caption_file = join(data_dir, 'faces/captions/'+loaded_data['image_list'][idx]+'.txt')
		with open(caption_file, 'r') as f:
			c = f.read().split('\n')[random_caption]
			c = word2vec.embed_sentence(c)
		captions[cnt,:] = c[0:caption_vector_length]
		image_files.append( image_file )
		cnt += 1

	z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])
	return real_images, wrong_images, captions, z_noise, image_files

if __name__ == '__main__':
	main()
