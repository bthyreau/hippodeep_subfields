#!/bin/bash
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

scriptpath=$(dirname $0)
if [ ${scriptpath:0:1} == '.' ]; then
	scriptpath=$PWD/$scriptpath
fi;

if [ "$1" == "" ]; then
    echo "Usage: $0 t1_mri_filename"
    exit 1;
fi

a=$1
ba=$(basename $a)
a=$(basename $a .gz)
a=$(basename $a .nii)
a=$(basename $a .img)
a=$(basename $a .hdr)
a=$(basename $a .mgz)
a=$(basename $a .mgh)
pth=$(dirname $1)

which antsApplyTransforms > /dev/null
if [ $? -eq "1" ]; then echo "ANTs scripts not in path"; exit; fi

cd $pth

if [ ! -f ${a}_mni0Affine.txt ]; then
	python $scriptpath/model_apply_head_and_cortex.py $ba
else
	echo "Reusing affine from existing file ${a}_mni0Affine.txt"
fi

# Note that the segmentation model can work with both affine and rigid registrations input
antsApplyTransforms -i ${ba} -r ${scriptpath}/hippoboxL_128.nii.gz -t ${a}_mni0Affine.txt -o ${a}_boxL.nii.gz --float
antsApplyTransforms -i ${ba} -r ${scriptpath}/hippoboxR_128.nii.gz -t ${a}_mni0Affine.txt -o ${a}_boxR.nii.gz --float

python $scriptpath/apply_hipposub.py ${a}_boxL.nii.gz

antsApplyTransforms -i ${a}_boxLhippo.nii.gz -r $ba -t [ ${a}_mni0Affine.txt,1] -o ${a}_hippoL_native.nii.gz --float -n MultiLabel[0.1]
antsApplyTransforms -i ${a}_boxRhippo.nii.gz -r $ba -t [ ${a}_mni0Affine.txt,1] -o ${a}_hippoR_native.nii.gz --float -n MultiLabel[0.1]

/bin/rm  ${a}_boxL.nii.gz ${a}_boxR.nii.gz
#/bin/rm  ${a}_boxLhippo.nii.gz ${a}_boxRhippo.nii.gz # always keep hippo space
echo "fslview $1 $pth/${a}_hippoL_native.nii.gz -l Random-Rainbow $pth/${a}_hippoR_native.nii.gz -l Random-Rainbow"
