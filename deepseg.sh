#!/bin/bash
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export OPENBLAS_NUM_THREADS=1

scriptpath=$(dirname $0); [ "${0:0:1}" != '/' ] && scriptpath="$PWD/$scriptpath"

while (( "$#" )); do
        case $1 in
        -d) DEBUG=1;;
        -m) MERGE=1;;
        -p) SAVEPROB="-saveprob";;
        -h) echo "Usage  : $0 [ -d ] [ -p ] t1_mri_image

Options:
 -d   : keep the MNI-space, higher-resolution copy of the input image
 -p   : output the full probabilistic map for each label
 -m   : (NOT IMPLEMENTED) merge left and right labels in a single output image."
        exit;;
        -*) echo "unexpected option $1"; exit;;
         *) if [ "$filename" != "" ] ; then echo "unexpected argument $1"; exit; fi; filename=$1;;
        esac
        shift
done

a=$filename
ba=$(basename $a)
for suffix in gz nii img hdr mgz mgh; do a=$(basename $a .$suffix); done
pth=$(dirname $filename)

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

python $scriptpath/apply_hipposub.py ${a}_boxL.nii.gz ${SAVEPROB}

antsApplyTransforms -i ${a}_boxL_hippo.nii.gz -r $ba -t [ ${a}_mni0Affine.txt,1] -o ${a}_hippoL_native.nii.gz --float -n MultiLabel[0.1]
antsApplyTransforms -i ${a}_boxR_hippo.nii.gz -r $ba -t [ ${a}_mni0Affine.txt,1] -o ${a}_hippoR_native.nii.gz --float -n MultiLabel[0.1]


if [ $DEBUG ]; then
    echo "fslview \"$pth/${a}_boxL.nii.gz\" \"$pth/${a}_boxL_hippo.nii.gz\" -t .5 -l Random-Rainbow &"
    echo "fslview \"$pth/${a}_boxR.nii.gz\" \"$pth/${a}_boxR_hippo.nii.gz\" -t .5 -l Random-Rainbow &"
else
    /bin/rm  ${a}_boxL.nii.gz ${a}_boxR.nii.gz
fi
#/bin/rm  ${a}_boxL_hippo.nii.gz ${a}_boxR_hippo.nii.gz # always keep hippo space
echo "fslview \"$filename\" \"$pth/${a}_hippoL_native.nii.gz\" -l Random-Rainbow \"$pth/${a}_hippoR_native.nii.gz\" -l Random-Rainbow"
