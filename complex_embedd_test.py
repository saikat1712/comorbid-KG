#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 18:10:07 2018

@author: saikat
"""

import downhill
import theano
import theano.tensor as TT
import scipy
import scipy.io
import numpy as np
import random
import sklearn
import sklearn.metrics
import logging
import operator
import json

import logging
logging.basicConfig()
import downhill.base
#downhill.base.logging.setLevel(20)
import uuid
import time
import subprocess
import sys,os
import colorsys
import models
from models import Complex_Model

start=time.time()

#################################################### batching.py ###########################################

import tools

cur_idx=0
class Batch_Loader(object):
	def __init__(self, train_triples, n_entities, batch_size=100, neg_ratio = 0.0, contiguous_sampling = False):
		self.train_triples = train_triples
		self.batch_size = batch_size
		self.n_entities = n_entities
		self.contiguous_sampling = contiguous_sampling
		self.neg_ratio = int(neg_ratio)
		self.idx = 0

		self.new_triples_indexes = np.empty((self.batch_size * (self.neg_ratio + 1) , 3), dtype=np.int64)
		self.new_triples_values = np.empty((self.batch_size * (self.neg_ratio + 1 )), dtype=np.float32)

	def __call__(self):

                global cur_idx
		if self.contiguous_sampling:
			if self.idx >= len(self.train_triples.values):
				self.idx = 0

			b = self.idx
			e = self.idx + self.batch_size
			this_batch_size = len(self.train_triples.values[b:e]) #Manage shorter batches (last ones)
			self.new_triples_indexes[:this_batch_size,:] = self.train_triples.indexes[b:e]
			self.new_triples_values[:this_batch_size] = self.train_triples.values[b:e]

			self.idx += this_batch_size

			last_idx = this_batch_size
		else:
			idxs =np.random.randint(0,len(self.train_triples.values),self.batch_size)
			self.new_triples_indexes[:self.batch_size,:] = self.train_triples.indexes[idxs,:]
			self.new_triples_values[:self.batch_size] = self.train_triples.values[idxs]

			last_idx = self.batch_size


		if self.neg_ratio > 0:

			#Pre-sample everything, faster
			rdm_entities = np.random.randint(0, self.n_entities, last_idx * self.neg_ratio)
			rdm_choices = np.random.random(last_idx * self.neg_ratio) < 0.5
			#Pre copying everyting
			self.new_triples_indexes[last_idx:(last_idx*(self.neg_ratio+1)),:] = np.tile(self.new_triples_indexes[:last_idx,:],(self.neg_ratio,1))
			self.new_triples_values[last_idx:(last_idx*(self.neg_ratio+1))] = np.tile(self.new_triples_values[:last_idx], self.neg_ratio)

			for i in range(last_idx):
				for j in range(self.neg_ratio):
					cur_idx = i* self.neg_ratio + j
					#Sample a random subject or object
                                         
					if rdm_choices[cur_idx]:
						self.new_triples_indexes[last_idx + cur_idx,0] = rdm_entities[cur_idx]
					else:
						self.new_triples_indexes[last_idx + cur_idx,2] = rdm_entities[cur_idx]

					self.new_triples_values[last_idx + cur_idx] = -1

			last_idx += cur_idx + 1

		train = [self.new_triples_values[:last_idx], self.new_triples_indexes[:last_idx,0], self.new_triples_indexes[:last_idx,1], self.new_triples_indexes[:last_idx,2]]


		return train



class TransE_Batch_Loader(Batch_Loader):
	#Hacky trick to normalize embeddings at each update
	def __init__(self, model, train_triples, n_entities, batch_size=100, neg_ratio = 0.0, contiguous_sampling = False):
		super(TransE_Batch_Loader, self).__init__(train_triples, n_entities, batch_size, neg_ratio, contiguous_sampling)

		self.model = model

	def __call__(self):
		train = super(TransE_Batch_Loader, self).__call__()
		train = train[1:]

		#Projection on L2 sphere before each batch
		self.model.e.set_value(tools.L2_proj(self.model.e.get_value(borrow = True)), borrow = True)

		return train



############################## evaluation.py #############################################################

class Result(object):
	"""
	Store one test results
	"""

	def __init__(self, preds, true_vals, ranks, raw_ranks):
		self.preds = preds
		self.ranks = ranks
		self.true_vals = true_vals
		self.raw_ranks = raw_ranks

		#Test if not all the prediction are the same, sometimes happens with overfitting,
		#and leads scikit-learn to output incorrect average precision (i.e ap=1)
		if not (preds == preds[0]).all() :
			#Due to the use of np.isclose in sklearn.metrics.ranking._binary_clf_curve (called by following metrics function),
			#I have to rescale the predictions if they are too small:
			preds_rescaled = preds
             

			diffs = np.diff(np.sort(preds))
			min_diff = min(abs(diffs[np.nonzero(diffs)]))
			if min_diff < 1e-8 : #Default value of absolute tolerance of np.isclose
				preds_rescaled = (preds * ( 1e-7 / min_diff )).astype('d')

			self.ap = sklearn.metrics.average_precision_score(true_vals,preds_rescaled)
			self.precision, self.recall, self.thresholds = sklearn.metrics.precision_recall_curve(true_vals,preds_rescaled) 
		else:
			logger.warning("All prediction scores are equal, probable overfitting, replacing scores by random scores")
			self.ap = (true_vals == 1).sum() / float(len(true_vals))
			self.thresholds = preds[0]
			self.precision = (true_vals == 1).sum() / float(len(true_vals))
			self.recall = 0.5
		
		
		self.mrr =-1
		self.raw_mrr =-1

		if ranks is not None:
			self.mrr = np.mean(1.0 / ranks)
			self.raw_mrr = np.mean(1.0 / raw_ranks)  




class CV_Results(object):
    
    """
    class that stores predictions and scores by indexing them by model, embedding_size and lmbda
    
    """
    def __init__(self):
		self.res = {}
		self.nb_params_used = {} #Indexed by model_s and embedding sizes, in order to plot with respect to the number of parameters of the model
        
    def add_res(self, res, model_s, embedding_size, lmbda, nb_params):
		if model_s not in self.res:
			self.res[model_s] = {}
		if embedding_size not in self.res[model_s]:
			self.res[model_s][embedding_size] = {}
		if lmbda not in self.res[model_s][embedding_size]:
			self.res[model_s][embedding_size][lmbda] = []

		self.res[model_s][embedding_size][lmbda].append( res )

		if model_s not in self.nb_params_used:
			self.nb_params_used[model_s] = {}
		self.nb_params_used[model_s][embedding_size] = nb_params
                     
    def extract_sub_scores(self, idxs):
        
        """
        Returns a new CV_Results object with scores only at the given indexes
        """
        new_cv_res = CV_Results()
        
        for j, (model_s, cur_res) in enumerate(self.res.items()):
			for i,(k, lmbdas) in enumerate(cur_res.items()):
				for lmbda, res_list in lmbdas.items():
					for res in res_list:
						if res.ranks is not None:
							#Concat idxs on ranks as subject and object ranks are concatenated in a twice larger array
							res = Result(res.preds[idxs], res.true_vals[idxs], res.ranks[np.concatenate((idxs,idxs))], res.raw_ranks[np.concatenate((idxs,idxs))])
						else:
							res = Result(res.preds[idxs], res.true_vals[idxs], None, None)
						
						new_cv_res.add_res(res, model_s, k, lmbda, self.nb_params_used[model_s][k])
        return new_cv_res
    
    
    def _get_best_mean_ap(self, model_s, embedding_size):
        """
        Averaging runs for each regularization value, and picking the best AP
        """
        lmbdas = self.res[model_s][embedding_size]
        
        mean_aps = []
        var_aps = []
        for lmbda_aps in lmbdas.values():
			mean_aps.append( np.mean( [ result.ap for result in lmbda_aps] ) )
			var_aps.append( np.std( [ result.ap for result in lmbda_aps] ) )
        cur_aps_moments = zip(mean_aps, var_aps)
        
        return max(cur_aps_moments, key = operator.itemgetter(0)) #max by mean
        
            
    
    def print_MRR_and_hits_given_params(self, model_s, rank, lmbda):
        
                
        mrr = np.mean( [ res.mrr for res in self.res[model_s][rank][lmbda] ] )
        raw_mrr = np.mean( [ res.raw_mrr for res in self.res[model_s][rank][lmbda] ] )
        
        ranks_list = [ res.ranks for res in self.res[model_s][rank][lmbda]]
        
        hits_at1 = np.mean( [ (np.sum(ranks <= 1) + 1e-10) / float(len(ranks)) for ranks in ranks_list] )
        hits_at3 = np.mean( [ (np.sum(ranks <= 3) + 1e-10) / float(len(ranks)) for ranks in ranks_list] )
        hits_at10= np.mean( [ (np.sum(ranks <= 10) + 1e-10) / float(len(ranks))  for ranks in ranks_list] )
        
        logger.info("%s\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%i\t%f" %(model_s, mrr, raw_mrr, hits_at1, hits_at3, hits_at10, rank, lmbda))
        
        out = open("/home/ksrao/Saikat/complex/result_out.txt",'a')
        out.write("%s\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%i\t%f\n" %(model_s, mrr, raw_mrr, hits_at1, hits_at3, hits_at10, rank, lmbda))
        
        out.close() 
        
#        
        return ( mrr, raw_mrr, hits_at1, hits_at3, hits_at10)
    
    
    def print_MRR_and_hits(self):
        
        metrics = {}
        logger.info("Model\t\t\tMRR\tRMRR\tH@1\tH@3\tH@10\trank\tlmbda")
        
        for j, (model_s, cur_res) in enumerate(self.res.items()):
            best_mrr = -1.0
            for i,(k, lmbdas) in enumerate(cur_res.items()):
                mrrs = []
                for lmbda, res_list in lmbdas.items():
                    mrrs.append( (lmbda, np.mean( [ result.mrr for result in res_list] ), np.mean( [ result.raw_mrr for result in res_list] ) ) )
                lmbda_mrr = max(mrrs, key = operator.itemgetter(1))
                mrr = lmbda_mrr[1]
                if mrr > best_mrr:
                    best_mrr = mrr
                    best_raw_mrr = lmbda_mrr[2]
                    best_lambda = lmbda_mrr[0]
                    best_rank = k
                    
            
                    
            metrics[model_s] = (best_rank, best_lambda) + self.print_MRR_and_hits_given_params(model_s, best_rank, best_lambda)
            open("/home/ksrao/Saikat/complex/result_out.txt", 'w').close()            
            
        return metrics
    


class Scorer(object):

	def __init__(self, train, valid, test, compute_ranking_scores = False,):

		self.compute_ranking_scores = compute_ranking_scores

		self.known_obj_triples = {}
		self.known_sub_triples = {}
		if self.compute_ranking_scores:
			self.update_known_triples_dicts(train.indexes)
			self.update_known_triples_dicts(test.indexes)
			if valid is not None:
				self.update_known_triples_dicts(valid.indexes)


	def update_known_triples_dicts(self,triples):
		for i,j,k in triples:
			if (i,j) not in self.known_obj_triples:
				self.known_obj_triples[(i,j)] = [k]
			elif k not in self.known_obj_triples[(i,j)]:
				self.known_obj_triples[(i,j)].append(k)

			if (j,k) not in self.known_sub_triples:
				self.known_sub_triples[(j,k)] = [i]
			elif i not in self.known_sub_triples[(j,k)]:
				self.known_sub_triples[(j,k)].append(i)
                
    	def compute_scores(self, model, model_s, params, eval_set):
            
            preds = model.predict(eval_set.indexes)
            
            ranks = None
            raw_ranks = None
            
            if self.compute_ranking_scores:
                #Then we compute the rank of each test:
                nb_test = len( eval_set.values) #1000
                ranks = np.empty( 2 * nb_test)
                raw_ranks = np.empty(2 * nb_test)
                
                if model_s.startswith("Complex") :
                    #Fast super-ugly filtered metrics computation for Complex
                    logger.info("Fast MRRs")
                    
                    def complex_eval_o(i,j):
                        return (e1[i,:] * r1[j,:]).dot(e1.T) + (e2[i,:] * r1[j,:]).dot(e2.T) + (e1[i,:] * r2[j,:]).dot(e2.T) - (e2[i,:] * r2[j,:]).dot(e1.T)
                    def complex_eval_s(j,k):
                        return e1.dot(r1[j,:] * e1[k,:]) + e2.dot(r1[j,:] * e2[k,:]) + e1.dot(r2[j,:] * e2[k,:]) - e2.dot(r2[j,:] * e1[k,:])
                    
                    if model_s.startswith("Complex"):
                        e1 = model.e1.get_value(borrow=True)
                        r1 = model.r1.get_value(borrow=True)
                        e2 = model.e2.get_value(borrow=True)
                        r2 = model.r2.get_value(borrow=True)
                        eval_o = complex_eval_o
                        eval_s = complex_eval_s
                        
                else:
                    #Generic version to compute ranks given any model:
                    logger.info("Slow MRRs")
                    n_ent = max(model.n,model.l)
                    idx_obj_mat = np.empty((n_ent,3), dtype=np.int64)
                    idx_sub_mat = np.empty((n_ent,3), dtype=np.int64)
                    idx_obj_mat[:,2] = np.arange(n_ent)
                    idx_sub_mat[:,0] = np.arange(n_ent)
                    
                    def generic_eval_o(i,j):
                        idx_obj_mat[:,:2] = np.tile((i,j),(n_ent,1))
                        return model.predict(idx_obj_mat)
                    def generic_eval_s(j,k):
                        idx_sub_mat[:,1:] = np.tile((j,k),(n_ent,1))
                        return model.predict(idx_sub_mat)
                    
                    eval_o = generic_eval_o
                    eval_s = generic_eval_s
                    
                for a,(i,j,k) in enumerate(eval_set.indexes[:nb_test,:]):
                    #Computing objects ranks
                    res_obj = eval_o(i,j)
                    raw_ranks[a] = 1 + np.sum( res_obj > res_obj[k] )
                    ranks[a] = raw_ranks[a] -  np.sum( res_obj[self.known_obj_triples[(i,j)]] > res_obj[k] )
                    #Computing subjects ranks
                    res_sub = eval_s(j,k)
                    raw_ranks[nb_test + a] = 1 + np.sum( res_sub > res_sub[i] )
                    ranks[nb_test + a] = raw_ranks[nb_test + a] - np.sum( res_sub[self.known_sub_triples[(j,k)]] > res_sub[i] )
                    
            return Result(preds, eval_set.values, ranks, raw_ranks) 
        
                

######################################################### experiment.py #########################################
class Experiment(object):
    
    def __init__(self, name,train, valid, test, positives_only = False,  compute_ranking_scores = False, entities_dict = None, relations_dict =None) :
        """
        An experiment is defined by its train and test set, which are two Triplets_set objects.
        """
        self.name = name
        self.train = train
        self.valid = valid
        self.test = test
        self.train_tensor = None
        self.train_mask = None
        self.positives_only = positives_only
        self.entities_dict = entities_dict
        self.relations_dict = relations_dict
        
        if valid is not None:
            self.n_entities = len(np.unique(np.concatenate((train.indexes[:,0], train.indexes[:,2], valid.indexes[:,0], valid.indexes[:,2], test.indexes[:,0], test.indexes[:,2]))))
            self.n_relations = len(np.unique(np.concatenate((train.indexes[:,1], valid.indexes[:,1], test.indexes[:,1]))))
        else:
            self.n_entities = len(np.unique(np.concatenate((train.indexes[:,0], train.indexes[:,2], test.indexes[:,0], test.indexes[:,2]))))
            self.n_relations = len(np.unique(np.concatenate((train.indexes[:,1], test.indexes[:,1]))))
        
#        tools.logger.info("Nb entities: " + str(self.n_entities))
#        tools.logger.info( "Nb relations: " + str(self.n_relations))
#        tools.logger.info( "Nb obs triples: " + str(train.indexes.shape[0]))
        
        self.scorer = Scorer(train, valid, test, compute_ranking_scores)
        #The trained models are stored indexed by name
        self.models = {}
        #The test Results are stored indexed by model name
        self.valid_results = CV_Results()
        self.results = CV_Results()
        
    def grid_search_on_all_models(self, params, embedding_size_grid = [1,2,3,4,5,6,7,8,9,10], lmbda_grid = [0.1], nb_runs = 10):
        """
        Here params is a dictionnary of Parameters, indexed by the names of each model, that
        must match with the model class names
        """
        #Clear previous results:
        self.results = CV_Results()
        self.valid_results = CV_Results()
        
        for model_s in params:
            #logger.info("Starting grid search on: " + model_s)
            #Getting train and test function using model string id:
#            print(model_s)
#            print(params)
            cur_params = params[model_s]
#            print(cur_params)
            for embedding_size in embedding_size_grid:
                for lmbda in lmbda_grid:
                    cur_params.embedding_size = embedding_size
                    cur_params.lmbda = lmbda
                    for run in range(nb_runs):
                        self.run_model(model_s,cur_params)
                        self.test_model(model_s)
                        
    def run_model(self,model_s,params):
        """
        Generic training for any model, model_s is the class name of the model class defined in module models
        """
        #Reuse ancient model if already exist:
        if model_s in self.models:
            model = self.models[model_s][0]

            
        else: #Else construct it:
            model = vars(models)[model_s]()

            
            
        self.models[model_s] = (model, params)
        
        model.fit(self.train, self.valid, params, self.n_entities, self.n_relations, self.n_entities, self.scorer)

    
    def test_model(self, model_s):
        
        """
        Generic testing for any model, model_s is the class name of the model class defined in module models
        """
        model, params = self.models[model_s]
        

        
        if self.valid is not None:
            res = self.scorer.compute_scores(model, model_s, params, self.valid)

            self.valid_results.add_res(res, model_s, params.embedding_size, params.lmbda, model.nb_params)
            
        res = self.scorer.compute_scores(model, model_s, params, self.test)

        self.results.add_res(res, model_s, params.embedding_size, params.lmbda, model.nb_params)
        
    def print_best_MRR_and_hits(self):
        """
        Print best results on validation set, and corresponding scores (with same hyper params) on test set
        
        """
#        tools.logger.info( "Validation metrics:")
        metrics = self.valid_results.print_MRR_and_hits()
        tools.logger.info( "Corresponding Test metrics:")
        for model_s, (best_rank, best_lambda, _,_,_,_,_) in metrics.items():
            self.results.print_MRR_and_hits_given_params(model_s, best_rank, best_lambda)
            
    def print_best_MRR_and_hits_per_rel(self):
        """
        Print best results on validation set, and corresponding scores (with same hyper params) on test set
        """
        
#        tools.logger.info( "Validation metrics:")
        metrics = self.valid_results.print_MRR_and_hits()
        
        tools.logger.info( "Corresponding per relation Test metrics:" )
        with open("/home/ksrao/Saikat/complex/relation_test.txt",'w') as f:
            for rel_name, rel_idx in self.relations_dict.items():
                tools.logger.info( rel_name )
                this_rel_row_idxs = self.test.indexes[:,1] == rel_idx
                this_rel_test_indexes = self.test.indexes[ this_rel_row_idxs ,:]
                this_rel_test_values = self.test.values[ this_rel_row_idxs ]
                this_rel_set = tools.Triplets_set(this_rel_test_indexes,this_rel_test_values)
                f.write("%s\n"%rel_name)
                for model_s, (best_rank, best_lambda, _,_,_,_,_) in metrics.items():
                    rel_cv_results = self.results.extract_sub_scores( this_rel_row_idxs)
                    rel_cv_results.print_MRR_and_hits_given_params(model_s, best_rank, best_lambda)
#                                    



######################################## exp_generators.py #####################################################

def parse_line(filename, line,i):
	line = line.strip().split("\t")
	sub = line[0]
	rel = line[1]
	obj = line[2]
	val = 1

	return sub,obj,rel,val

def load_triples_from_txt(filenames, entities_indexes = None, relations_indexes = None, add_sameas_rel = False, parse_line = parse_line):
    
    """ 
    Take a list of file names and build the corresponding dictionary of triples
    """
    
    if entities_indexes is None:
        entities_indexes=dict()
        entities=set()
        next_ent=0
    else:
        entities=set(entities_indexes)
        next_ent=max(entities_indexes.values()) + 1
                    
    if relations_indexes is None:
        relations_indexes=dict()
        relations=set()
        next_rel=0
    else:
        relations=set(relations_indexes)
        next_rel=max(relations_indexes.values()) + 1
        
    data=dict()
    
    for filename in filenames:
        with open(filename) as f:
            lines=f.readlines()
            for i,line in enumerate(lines):
                sub,obj,rel,val=parse_line(filename,line,i)
                if sub in entities:
                    sub_ind=entities_indexes[sub]
                else:
                    sub_ind=next_ent
                    next_ent+=1
                    entities_indexes[sub]=sub_ind
                    entities.add(sub)
                if obj in entities:
                    obj_ind=entities_indexes[obj]
                else:
                    obj_ind=next_ent
                    next_ent+=1
                    entities_indexes[obj]=obj_ind
                    entities.add(obj)
                if rel in relations:
                    rel_ind=relations_indexes[rel]
                else:
                    rel_ind=next_rel
                    next_rel+=1
                    relations_indexes[rel]=rel_ind
                    relations.add(rel)
                data[(sub_ind,rel_ind,obj_ind)]=val                
        

	if add_sameas_rel :
		rel = "sameAs_"
		rel_ind = next_rel
		next_rel += 1
		relations_indexes[rel] = rel_ind
		relations.add(rel)
		for sub in entities_indexes:
			for obj in entities_indexes:
				if sub == obj:
					data[ (entities_indexes[sub], rel_ind, entities_indexes[obj])] = 1
				else:
					data[ (entities_indexes[sub], rel_ind, entities_indexes[obj])] = -1
   
    return data, entities_indexes, relations_indexes



def build_data(name, path='/home/ksrao/Saikat/complex/datasets/'):
    
    
    folder = path  + name + '/'
    
    train_triples, entities_indexes, relations_indexes = load_triples_from_txt([folder + 'train.txt'],add_sameas_rel = False, parse_line = parse_line)
    
    valid_triples, entities_indexes, relations_indexes =  load_triples_from_txt([folder + 'valid.txt'],entities_indexes = entities_indexes , relations_indexes = relations_indexes,	add_sameas_rel = False, parse_line = parse_line)
    
    test_triples, entities_indexes, relations_indexes =  load_triples_from_txt([folder + 'test.txt'],entities_indexes = entities_indexes, relations_indexes = relations_indexes,add_sameas_rel = False, parse_line = parse_line)
    
    train = tools.Triplets_set(np.array(list(train_triples.keys())), np.array(list(train_triples.values())))
    valid = tools.Triplets_set(np.array(list(valid_triples.keys())), np.array(list(valid_triples.values())))
    test = tools.Triplets_set(np.array(list(test_triples.keys())), np.array(list(test_triples.values())))
    
    with open("/home/ksrao/Saikat/complex/entity_dic.txt",'w') as f1:
        f1.write(json.dumps(entities_indexes))
    with open("/home/ksrao/Saikat/complex/relations_dic.txt",'w') as f1:
        f1.write(json.dumps(relations_indexes))
    
    return Experiment(name,train, valid, test, positives_only = True, compute_ranking_scores = True, entities_dict = entities_indexes, relations_dict = relations_indexes)


#r1=Experiment(train, valid, test, positives_only = True, compute_ranking_scores = True, entities_dict = entities_indexes, relations_dict = relations_indexes)
#def load_mat_file(name, path, matname, load_zeros = False, prop_valid_set = .1, prop_test_set=0):
#
#	x = scipy.io.loadmat(path + name)[matname]
#
#
#	if sp.issparse(x): 
#		if not load_zeros:
#			idxs = x.nonzero()
#
#			indexes = np.array(zip(idxs[0], np.zeros_like(idxs[0]), idxs[1]))
#			np.random.shuffle(indexes)
#
#			nb = indexes.shape[0]
#			i_valid = int(nb - nb*prop_valid_set - nb * prop_test_set)
#			i_test = i_valid + int( nb*prop_valid_set)
#
#			train = Triplets_set(indexes[:i_valid,:], np.ones(i_valid))
#			valid = Triplets_set(indexes[i_valid:i_test,:], np.ones(i_test - i_valid))
#			test = Triplets_set(indexes[i_test:,:], np.ones(nb - i_test))
#
#
#	return Experiment(name,train, valid, test, positives_only = True, compute_ranking_scores = True)
	
#################################### Main calling program ##################################################

if __name__ =="__main__": 
    #Load data, ensure that data is at path: 'path'/'name'/[train|valid|test].txt
    doc =build_data(name = 'kg',path = '/home/ksrao/Saikat/complex/datasets/')
    params = tools.Parameters(learning_rate = 0.5,max_iter = 1000,batch_size = int(len(doc.train.values) / 100),neg_ratio = 10, 
						valid_scores_every = 50,
						learning_rate_policy = 'adagrad',
						contiguous_sampling = False )
    all_params = { "Complex_Logistic_Model" : params } ; emb_size = 200; lmbda =0.01;



    doc.grid_search_on_all_models(all_params, embedding_size_grid = [emb_size], lmbda_grid = [lmbda], nb_runs = 1)
   
    
    
    doc.print_best_MRR_and_hits()
    
    doc.print_best_MRR_and_hits_per_rel()
    
    temp1=[]
    temp2=[]
    f1 = open("/home/ksrao/Saikat/complex/result_out.txt",'rb')
    f2= open("/home/ksrao/Saikat/complex/relation_test.txt",'rb')
    for line in f1:
        temp1.append(line)
    for line in f2:
        temp2.append(line)
        
    f1.close()
    f2.close()
        
    with open("/home/ksrao/Saikat/complex/result_per_test_relation.txt",'w') as out:
        
        for k1,k2 in zip(temp2,temp1):
            out.write("%s\t%s\n"%(k1,k2))

    end=time.time()
    print(end-start)
            
    #Save ComplEx embeddings (last trained model, not best on grid search if multiple embedding sizes and lambdas)
    e1 = doc.models["Complex_Logistic_Model"][0].e1.get_value(borrow=True)
    e2 = doc.models["Complex_Logistic_Model"][0].e2.get_value(borrow=True)
    r1 = doc.models["Complex_Logistic_Model"][0].r1.get_value(borrow=True)
    r2 = doc.models["Complex_Logistic_Model"][0].r2.get_value(borrow=True)
    scipy.io.savemat('Complex_Logistic_Model.mat', \
			{'entities_real' : e1, 'relations_real' : r1, 'entities_imag' : e2, 'relations_imag' : r2  })
