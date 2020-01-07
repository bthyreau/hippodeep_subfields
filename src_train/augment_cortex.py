# This scripts generate some random nonlinear deformation on input nifti image
# It is based on ANTs and requires antsApplyTransforms
# Depends on scipy and nibabel
#
# It simply generates random gaussian blobs for a warping fields, possibly along masks,
# then it applies this warping to the input image, and corresponding labels images (if any)
# There is two steps of deformation: a few large ones and many small ones
#
# There are a lot of tweakable parameters which must be adjusted by trial-and-error
# for any new type of input, see the "TWEAK" keyword in the code
#
#
# It should be run from the data directory, with first arg is the original image .nii.gz
# For mask and labels, it is currently deduced from the filename, tweakable below

# 2019-12-31 Hack for hippo

import sys, time
t1= time.time()
import nibabel, os
import scipy.ndimage
from numpy import *
import numpy as np
from numpy.random import randint, random_sample
import sys

fname = sys.argv[1]
assert not "/" in fname
assert fname.endswith(".nii.gz")

#subjid = "example_brain_t1.nii.gz".replace(".nii.gz", "")
subjid = fname.replace(".nii.gz", "")

orig = nibabel.load(subjid + ".nii.gz")
affineh = orig.affine
sh = orig.shape[:3] 

#labfname = "convseg/example_brain_subj_hemi_L.nii.gz"
#labfname = "example_brain_mask.nii.gz"
#labfname = ""
labfname = fname.replace("_orig_box", "_hippoAmyg_box")
venfname = fname.replace("_orig_box", "_ventricles_box")
print(labfname)

# TWEAK : use some optional masks to limit where distortions seed can occurs
# wseedmap1 is a list of candidate voxels used for big-deformation, wseedmap2 for small deformations

if not os.path.exists(labfname):
    # no mask provided, pick a regular grid of voxels
    wseedmap1 = wseedmap2 = [x*4 for x in np.where(ones([i//4 for i in sh]))]
    labfname = None
else:
    labimg = nibabel.load(labfname)
    ventimg = nibabel.load(venfname)
    if 1:
        # take some subset (1/7th) of the labelled voxels, and add the ventricles voxels
        seedmap = labimg.get_data() > .1
        hippo_latvent_map = ventimg.get_data() == 2
        wseedmap2 = np.where(seedmap)
        wseedmap3 = np.where(hippo_latvent_map)
        perm = np.random.permutation(len(wseedmap2[0]))[::7]
        wseedmap2 = tuple(np.hstack([np.vstack(wseedmap2)[:,perm], np.vstack(wseedmap3)]))        

        hippo_vent_map = ventimg.get_data() == 1
        wseedmap1 = tuple(np.hstack([[x*4 for x in np.where(np.ones([i//4 for i in sh]))] , np.where(hippo_vent_map)]))

    if 0:
        # TWEAK instead of using the same brain mask, or a different mask (e.g. a ribbon) for
        # small deformations, we may also want to use the border (bool laplacian) of the brain mask
        seedmap = scipy.ndimage.laplace(labimg.get_data()) != 0
        wseedmap2 = np.where(seedmap) # small-deformation

print("Running %s, with mask=%s" % (subjid, labfname))

if not os.path.exists("generated"):
    os.mkdir("generated")

headerh = orig.header
headerh["intent_code"] = 1007
headerh["datatype"] = 16

need_flip_axis = nibabel.orientations.io_orientation(orig.affine)[:,1] * np.array([-1, -1, 1])

for numid in range(0, 5): # TWEAK: how many image to generate ? (also affects output filename)
    t1= time.time()
    prefix = "g_%s_v%04d" % (subjid, numid)

    def make_kernel():
        (d1,d2,d3) = (8, 8, 8)
        v = zeros((60, 60, 60))
        v[30,30,30] = 300
        vv=scipy.ndimage.gaussian_filter(v, (d1,d2,d3), truncate=6)
        vv /= vv.max()
        return vv

    def make_dkernel(vv, axis, order=1):
        v = vv.copy()
        dv = diff(v, order, axis)
        if axis == 0:
            v[order:,:,:] = dv / dv.max()
        elif axis == 1:
            v[:,order:,:] = dv / dv.max()
        elif axis == 2:
            v[:,:,order:] = dv / dv.max()
        return v

    kernel = make_kernel()
    kernel0_0 = make_dkernel(kernel, 0) * need_flip_axis[0]
    kernel0_1 = make_dkernel(kernel, 1) * need_flip_axis[1]
    kernel0_2 = make_dkernel(kernel, 2) * need_flip_axis[2]
    kernel2_s = [x[::2,::2,::2][2:-2,2:-2,2:-2] for x in (kernel0_0, kernel0_1, kernel0_2)]
    kernel1_s = [x[8:-8,8:-8,8:-8] for x in (kernel0_0, kernel0_1, kernel0_2)] # TWEAK this function if large anisotropy.
    #kernel1 = kernel0[8:-8,8:-8,8:-8]

    def inpaint3d(p, kernel, canvas):
        canvas[int(p[0]-kernel.shape[0]/2):, int(p[1]-kernel.shape[1]/2):, int(p[2]-kernel.shape[2]/2):][:int(kernel.shape[0]),:int(kernel.shape[1]),:int(kernel.shape[2])] += kernel

    def inpaint2d(p, kernel, canvas):
        canvas[p[0]-kernel.shape[0]/2:, p[1]-kernel.shape[1]/2:][:kernel.shape[0],:kernel.shape[1]] += kernel



    # Big Canvas (for large-magnitude distortions)
    ##########################################
    canvasx = zeros(sh)
    canvasy = zeros(sh)
    canvasz = zeros(sh)

    if 1: # TWEAK: do?
        ws = np.random.randint(0, len(wseedmap1[0]), size=6) # TWEAK: size=How many large deformation blob
        for p in ws:
            scales_xyz = 2 * np.random.normal(loc=.5, scale=1, size=3) # >0 expansion; <0 shrinking
            for k, s, c in zip(kernel2_s, scales_xyz, [canvasx, canvasy, canvasz]):
                x = np.clip(wseedmap1[0][p], shape(k)[0]//2,  sh[0] - shape(k)[0]//2)
                y = np.clip(wseedmap1[1][p], shape(k)[1]//2,  sh[1] - shape(k)[1]//2)
                z = np.clip(wseedmap1[2][p], shape(k)[2]//2,  sh[2] - shape(k)[2]//2)
                inpaint3d((x,y,z), s * k, c)

        # Smooth the generated gaussian
        # TWEAK: magnitude of warp effect (need to be adjusted by trial and error)
        # (with small kernel, use low mul_fact; ((50, (12,12,12)); or (200, (16,16,16)))
        mul_fact = 150.
        kern = (15, 15, 15) # lower more distorted (5..25)
        canvasx = scipy.ndimage.gaussian_filter(canvasx, kern[0], truncate=4)*mul_fact
        canvasy = scipy.ndimage.gaussian_filter(canvasy, kern[1], truncate=4)*mul_fact
        canvasz = scipy.ndimage.gaussian_filter(canvasz, kern[2], truncate=4)*mul_fact

    big_canvas = canvasx, canvasy, canvasz


    # Small Canvas (for more local distortions)
    ##########################################
    canvasx = zeros(sh)
    canvasy = zeros(sh)
    canvasz = zeros(sh)

    if 1: # TWEAK: do?
        ws = np.random.randint(0, len(wseedmap2[0]), size=20) # TWEAK: size=How many small deformation blob
        for p in ws:
            scales_xyz = np.random.normal(loc=0, scale=1, size=3) # >0 expansion; <0 shrinking
            for k, s, c in zip(kernel2_s, scales_xyz, [canvasx, canvasy, canvasz]):
                x = np.clip(wseedmap2[0][p], shape(k)[0]//2,  sh[0] - shape(k)[0]//2)
                y = np.clip(wseedmap2[1][p], shape(k)[1]//2,  sh[1] - shape(k)[1]//2)
                z = np.clip(wseedmap2[2][p], shape(k)[2]//2,  sh[2] - shape(k)[2]//2)
                inpaint3d((x,y,z), s * k, c)

        # TWEAK: magnitude of warp effect (need to be adjusted by trial and error)
        mul_fact = 9.
        kern = 8 # the lower, the more distorted
        canvasx = scipy.ndimage.gaussian_filter(canvasx, kern, truncate=3)*mul_fact
        canvasy = scipy.ndimage.gaussian_filter(canvasy, kern, truncate=3)*mul_fact
        canvasz = scipy.ndimage.gaussian_filter(canvasz, kern, truncate=3)*mul_fact

    small_canvas = canvasx, canvasy, canvasz

    canvasx, canvasy, canvasz = big_canvas[0] + small_canvas[0],big_canvas[1] + small_canvas[1],  big_canvas[2] + small_canvas[2]


    if 1:
        # TWEAK:
        canvasx += random_sample(size=1) * 4 - 2  # move up to -2..+2 pixels in x direction
        canvasy += random_sample(size=1) * 4 - 2
        canvasz += random_sample(size=1) * 4 - 2

    # Write the output warping field, to be used with the ANTS command below
    v = zeros(sh + (1,3,), np.float32)
    v[:,:,:,0,0] = canvasx
    v[:,:,:,0,1] = canvasy
    v[:,:,:,0,2] = canvasz
    nibabel.Nifti1Image(v, affineh, headerh).to_filename("%s_replop1Warp.nii" % prefix)

    if 1: # TWEAK
        # Augmentation with rotation/zoom;
        # This is done by writting a separate affine matrix
        ##########################################

        def Rx(alpha):
            R = np.identity(4, float32)
            R[1,1] = cos(alpha)
            R[1,2] = sin(alpha)
            R[2,1] = -sin(alpha)
            R[2,2] = cos(alpha)
            return R
        def Ry(alpha):
            R = np.identity(4, float32)
            R[0,0] = cos(alpha)
            R[0,2] = -sin(alpha)
            R[2,0] = sin(alpha)
            R[2,2] = cos(alpha)
            return R
        def Rz(alpha):
            R = np.identity(4, float32)
            R[0,0] = cos(alpha)
            R[0,1] = sin(alpha)
            R[1,0] = -sin(alpha)
            R[1,1] = cos(alpha)
            return R

        zooms = array([1.,1.,1.,1.])
        zooms[:3] += np.random.normal(scale=.1, size=3) * .4 # TWEAK axis scaling
        angles = np.random.normal(scale=pi/32., size=3) * 0.25 # (.05, .05, .05) # TWEAK: amount of rotation motion on x,y,z
        rotation_matrix = dot(  dot( dot(Rx(angles[0]), diag(zooms)), Ry(angles[1]) ), Rz(angles[2])  )

        center_of_rotation_mm = dot(affineh, [x/2. for x in sh] + [1])[:-1] # pick center of the box for rotation center
        center_of_rotation_mm += np.random.normal(scale=1, size=3) * (1., 1., 1.)

    else: # No affine-based augmentation
        center_of_rotation_mm, rotation_matrix = [0, 0, 0], np.identity(4)

    # Generate a ANTS-readable affine file
    antmattxt = '#Insight Transform File V1.0\n#Transform 0\nTransform: MatrixOffsetTransformBase_double_3_3\nParameters: %s\nFixedParameters: %s\n'
    ptxt = " ".join(["%4.6f" % x for x in rotation_matrix[:,:3].ravel()])
    ctxt = " ".join(["%4.6f" % x for x in center_of_rotation_mm])
    open("%s_replop0Affine.txt" % prefix,"w").write(antmattxt % ( ptxt, ctxt))



    # Apply the warping fields and matrix by actually calling ANTS
    #####################

    fake_background = "-v %d" % np.random.uniform(0, 100) # TWEAK, although not useful
    # The ANTS command
    # warp the main image:
    os.system ("antsApplyTransforms -d 3 -i %s.nii.gz -r %s.nii.gz  -o generated/%s.nii.gz -t %s_replop1Warp.nii -t %s_replop0Affine.txt --float %s > /dev/null" % (subjid, subjid, prefix, prefix, prefix, fake_background))

    if 1: # TWEAK: warp other similar images, such as labels ?
        # TWEAK: need to adjust the filename (-i, -o) of the labels and target
        # TWEAK: need to adjust the resampling of targets (eg. Linear, or NearestNeighbor, MultiLabels (slow!))
        os.system ("antsApplyTransforms -d 3 -i %s.nii.gz -r %s.nii.gz  -o generated/%s_labels.nii.gz -t %s_replop1Warp.nii -t %s_replop0Affine.txt --float -n MultiLabel[0.001] > /dev/null" % (subjid.replace("orig", "hippoAmyg"), subjid, prefix, prefix, prefix))
        os.system("fslmaths generated/%s_labels.nii.gz generated/%s_labels.nii.gz -odt char" % (prefix,prefix))

        # Not sure i need it, because these ventricles are quite voxelized/aliased
        os.system ("antsApplyTransforms -d 3 -i %s.nii.gz -r %s.nii.gz  -o generated/%s_ventricles.nii.gz -t %s_replop1Warp.nii -t %s_replop0Affine.txt --float -n NearestNeighbor > /dev/null" % (subjid.replace("orig", "ventricles"), subjid, prefix, prefix, prefix))
        os.system("fslmaths generated/%s_ventricles.nii.gz generated/%s_ventricles.nii.gz -odt char" % (prefix,prefix))

    if 1:
        # Alter contrast
        # This reloads the output file above, hack its color and save
        ##########################################
        from scipy.interpolate import interp1d
        canvasi = zeros(sh)
        for _ in range(5):
            x, y, z = randint(30, sh[0]-1-30), randint(30, sh[1]-1-30), randint(30, sh[2]-1-30)
            s = random_sample(size=1) * 2 - 1
            inpaint3d((x,y,z), s * kernel0_0, canvasi)

        canvasi=clip((scipy.ndimage.gaussian_filter(canvasi, (12,12,12), truncate=2)*2. + 1), .9, 1.05)

        imgout = nibabel.load("generated/%s.nii.gz" % prefix)

        def fuzz_histo(d):
            nbstep = 4
            s = np.clip(random_sample(size=nbstep + 4), 0.35, .75)
            s[:3] = [0, .5, .5]
            s[-1] = .5
            v = cumsum(s)
            v /= v[-1]
            f = interp1d(linspace(d.min(), d.max()+1, len(v)), v, kind="cubic")
            return f(d.flat).reshape(d.shape)

        v=imgout.get_data()
        v = (fuzz_histo(v) * (v.max() - v.min())) + v.min()
        vv = scipy.ndimage.laplace (v) * random_sample() * .25 # was 1
        out=(canvasi*(v-vv))
        out[out < 1] = 0

        h = imgout.header
        h.set_data_dtype(uint8)
        nibabel.Nifti1Image(out, imgout.affine, h).to_filename("generated/%s_intens.nii.gz" % prefix)

        os.system("/bin/rm generated/%s.nii.gz" % prefix)
    
    if 1: # TWEAK to keep debug (especially ...1Warp.nii, which helps to visualize the warping blobs)
        os.system("/bin/rm -f %s_replop1Warp.nii" % prefix)
        os.system("/bin/rm -f %s_replop0Affine.txt" % prefix)

    print ("generated/%s.nii.gz %s" % (prefix, time.time()-t1))
