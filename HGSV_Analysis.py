#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 10:05:54 2019

@author: mitchnodzak

Revisions to ASE analysis based on python rather than command line tools.

"""
from pybedtools import BedTool
from SVtools.helperFunctions import gtfReader, VCFparser

# grep only passing variants from integrated VCF.
# saved as file name below...
# 'PASS_Illumina_Integrate_20170206.ALL.vcf' as infile.

infile = 'PASS_Illumina_Integrate_20170206.ALL.vcf'
svtype = infile.split('/')[-1].split('.')[1]
pref = '.'.join(infile.split('/')[-1].split('.')[0].split("_")[0:2])
outfy = pref+'.'+svtype+'_integrate.bed'


col_names =  ['#chrom', 'start', 'stop', 'REF', 'ALT', 'QUAL', 'FILT', 'sample', 'GT']
hgsv_list = ['HG00512','HG00513','HG00514', 'HG00731', 'HG00732', 'HG00733', 'NA19238', 'NA19239', 'NA19240']

# Extract the relevant information to bed file for safe keeping, and a pandas 
# DataFrame and generate a subsequent coordinate intersections 
# with genome annotations.



# x = BedTool.from_dataframe(df)

HGSV_integrated_VCF = '../PASS_Illumina_Integrate_20170206.ALL.vcf'

x = VCFparser(HGSV_integrated_VCF)

### grab the Bedtools corresponding to hgsv_list index.
for i in range(len(hgsv_list)):
    hgsv_list[i] = x.bedtoolsList[i]


gtf = '../gencode.v25.annotation.gtf'

gencodev25 = gtfReader(gtf,'gene', 'protein_coding')

gencode = BedTool.from_dataframe(gencodev25)

# Find where heterozygous deletions and duplications 100% overlap a gene.
SVoverlap100 = []
for i in range(len(hgsv_list)):
    overlap = str(hgsv_list[i])+"_100"
    overlap = gencode.intersect(hgsv_list[i], f=1.0,wa=True,wb=True)
    SVoverlap100.append(overlap)
             



                             