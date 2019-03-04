#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 13:06:56 2019

@author: mitchnodzak
"""
from SVtools.helperFunctions import gtfReader, VCFparser
import tadtool.tad as tad
import tadtool.plot as tp
from pybedtools import BedTool

# load regions data set
regions = tad.HicRegionFileReader().regions("chr12_20-35Mbregions.bed")

# load matrix
matrix = tad.HicMatrixFileReader().matrix("chr12_20-35Mb.matrix.txt")

# prepare plot
tad_plot = tp.TADtoolPlot(matrix, regions, norm='lin', max_dist=1000000, algorithm='directionality')
fig, axes = tad_plot.plot('chr12:31000000-34000000')

# show plot
import matplotlib.pyplot as plt
plt.show()

infile = 'PASS_Illumina_Integrate_20170206.ALL.vcf'
svtype = infile.split('/')[-1].split('.')[1]
pref = '.'.join(infile.split('/')[-1].split('.')[0].split("_")[0:2])
outfy1 = pref+'.'+svtype+'_integrate.TADanalysis.hetSites.bed'
outfy2 = pref+'.'+svtype+'_integrate.TADanalysis.all.bed'

col_names =  ['#chrom', 'start', 'stop', 'REF', 'ALT', 'QUAL', 'FILT', 'sample', 'GT']
hgsv_list = ['HG00512','HG00513','HG00514', 'HG00731', 'HG00732', 'HG00733', 'NA19238', 'NA19239', 'NA19240']

# First create list of bedtools objects, one for each sample's heterozygous variants.

# x = BedTool.from_dataframe(df)

HGSV_integrated_VCF = '../PASS_Illumina_Integrate_20170206.ALL.vcf'

x = VCFparser(HGSV_integrated_VCF, outfy1, "heterozygous")
y = VCFparser(HGSV_integrated_VCF, outfy2)

### grab the Bedtools corresponding to hgsv_list index.
for i in range(len(hgsv_list)):
    hgsv_list[i] = x.bedtoolsList[i]

gtf = '../gencode.v25.annotation.gtf'

gencodev25 = gtfReader(gtf,'gene', 'protein_coding')

gencode = BedTool.from_dataframe(gencodev25)



