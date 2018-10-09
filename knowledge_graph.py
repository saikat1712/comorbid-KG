#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 17:28:53 2018

@author: saikat
"""

from random import shuffle
import time


start=time.time()
relations=['_has_interaction','_has_function','_has_gene_phenotype','_causes','_involves','_disrupts','_treats','_has_side_effects','_has_disease_phenotype','_associated_to']
bi_relations=['_is_treated_by','_is_function_of','_side_effect_of','_is_gene_phenotype_of','_is_caused_by']

##################### Gene has_interaction to itself ####################################################
temp1=[]
with open("/home/saikat/Downloads/Knowledge_graph_proj/final_entrez_network.txt") as f:#,open("/home/saikat/Desktop/gene_to_gene.txt",'w') as out:
    for line in f:
        ind1=line.strip('\n').split('\t')
        temp1.append((ind1[0],relations[0],ind1[1]))
        temp1.append((ind1[1],relations[0],ind1[0]))
#        out.write("%s\t%s\t%s\n"%(ind1[0],relations[0],ind1[1]))

####################  Entrez gene ID "has_function" GO term ###############################

temp2=[]
with open("/home/saikat/Downloads/Knowledge_graph_proj/gene2go_human.txt") as f:
    for line in f:
        ind2=line.strip('\n').split('\t')
        temp2.append((ind2[1],relations[1],ind2[0]))
        temp2.append((ind2[0],bi_relations[1],ind2[1]))

##################### Entrez gene ID "has_gene_phenotype" GO term ##################

temp3=[]
with open("/home/saikat/Downloads/Knowledge_graph_proj/gene_to_hpo.txt") as f:
    for line in f:
        ind3=line.strip('\n').split('\t')
        temp3.append((ind3[0],relations[2],ind3[1]))
        temp3.append((ind3[1],bi_relations[3],ind3[0]))

######################  Entrez gene "has_association"/"causes" disease ###################

temp4=[]
with open("/home/saikat/Downloads/Knowledge_graph_proj/Bi_direction-rel_temp_updated/dis_gene.txt") as f:
    for line in f:
        ind4=line.strip('\n').split('\t')
        temp4.append((ind4[1],relations[3],ind4[0]))
        temp4.append((ind4[0],bi_relations[4],ind4[1]))

##################### Pathway involves Entrez_gene ######################

temp5=[]
with open("/home/saikat/Downloads/pathways/entrez_to_pathway.txt") as f:
    for line in f:
        ind5=line.strip('\n').split('\t')
        temp5.append((ind5[1],relations[4],ind5[0]))
        
        
##################### Disease disrupts pathway ############################

temp6=[]
with open("/home/saikat/Downloads/Knowledge_graph_proj/Bi_direction-rel_temp_updated/dis_pathway.txt") as f:
    for line in f:
        ind6=line.strip('\n').split('\t')
        temp6.append((ind6[0],relations[5],ind6[1]))
      
        
##################### Drug treats Disease ###################################

temp7=[]
with open("/home/saikat/Downloads/Knowledge_graph_proj/Bi_direction-rel_temp_updated/drug_to_icd.txt") as f:
    for line in f:
        ind7=line.strip('\n').split('\t')
        temp7.append((ind7[0],relations[6],ind7[1]))
        temp7.append((ind7[1],bi_relations[0],ind7[0]))
        
##################### Drug has_side_effects hpo #############################

temp8=[]
with open("/home/saikat/Downloads/Knowledge_graph_proj/drug_side_effects_to_hpo.txt") as f:
    for line in f:
        ind8=line.strip('\n').split('\t')
        temp8.append((ind8[0],relations[7],ind8[1]))
        temp8.append((ind8[1],bi_relations[2],ind8[0]))
        
##################### disease has_phenotype hpo ############################

temp9=[]
with open("/home/saikat/Downloads/Knowledge_graph_proj/Bi_direction-rel_temp_updated/icd_to_hpo.txt") as f:
    for line in f:
        ind9=line.strip('\n').split('\t')
        temp9.append((ind9[0],relations[8],ind9[1]))
       
        
###################### disease associated_to disease ########################

temp10=[]
with open("/home/saikat/Downloads/Knowledge_graph_proj/Bi_direction-rel_temp_updated/dis_dis.txt") as f:
    for line in f:
        ind10=line.strip('\n').split('\t')
        temp10.append((ind10[0],relations[9],ind10[1]))



############################################################################
        
all=temp1+temp2+temp3+temp4+temp5+temp6+temp7+temp8+temp9+temp10
shuffle(all)
with open("/home/saikat/Downloads/Knowledge_graph_proj/Bi_direction-rel_temp_updated/complete_knowledge_base_with_bi_relations_temp_stable_icd.txt",'w') as out:
    for a,b,c in all:
        out.write("%s\t%s\t%s\n"%(a,b,c))
#print(len(all)) 
end=time.time()       

print(end-start)
print("Completed")





        
    
