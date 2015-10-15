#!/usr/bin/env python
"""
Merges marker files from several substacks into a single file.
"""
import argparse
import math
from bcfind.volume import SubStack,Center
import numpy as np
from bcfind.markers import distance, match_markers, match_markers_with_icp
from bcfind.scripts.fuse_markers import do_fuse_with_icp,do_fuse
from bcfind import threshold
from os import listdir
from os.path import isfile, isdir, join, exists
from bcfind.utils import mkdir_p
from multiview.rigid_transformation import parse_transformation_file
import thresholds


def inside_margin(c,substack):
	m = substack.plist['Margin']/2
	return min(c.x-m,c.y-m,c.z-m,substack.info['Width']-m-c.x,substack.info['Height']-m-c.y,substack.info['Depth']-m-c.z)

def filter_outside_markers(C,substack):
	C_filtered=[]
        for c in C:
		if (c.x >=0 and c.x <= substack.info['Width'] -1 and c.y>=0 and c.y <= substack.info['Height']-1 and c.z>=0 and c.z <= substack.info['Depth']-1):
			C_filtered.append(c)
	return C_filtered

def compute_fusion(substack,C1,C2,max_distance, match_distance=None, do_icp=True, verbose=False):
    if do_icp:
        _, _, _,C_fused = do_fuse_with_icp(substack,C1,C2,max_distance, match_distance=match_distance, verbose=verbose)
    else:
        _, _, _,C_fused = do_fuse(substack,C1,C2,max_distance,verbose)

    return C_fused
        

def transform_markers(args,substack,C_moving,view_ids):
	view=view_ids[0]+'_'+view_ids[2]
	args.substack_id = substack.substack_id
        R,t=parse_transformation_file(join(args.dir_registration,args.substack_id,view))
	va=-math.asin(R[2,0]);vb=math.atan2(R[2,1]/math.cos(va),R[2,2]/math.cos(va));vc=math.atan2(R[1,0]/math.cos(va),R[1,1]/math.cos(va))
	eul_angles=np.array([va,vb,vc])
	C_moving_t=[]
	if np.linalg.norm(t)<args.norm_t and np.linalg.norm(eul_angles)<args.norm_r:
		t = -np.dot(R.T, t)
		R=R.T
		for c in C_moving:
			c_t=Center(R[0][0]*c.x + R[0][1]*c.y + R[0][2]*c.z + t[0],R[1][0]*c.x + R[1][1]*c.y + R[1][2]*c.z + t[1],R[2][0]*c.x + R[2][1]*c.y + R[2][2]*c.z + t[2])
			c_t.name='merged'
			c_t.hue=c.hue
			C_moving_t.append(c_t)
	return C_moving_t

def filter_valid_pairs(args,substack,list_views):
	args.substack_id = substack.substack_id
	valid_pairs={}
	for view in list_views:
		is_good = False
		if isfile(args.dir_registration+'/'+args.substack_id+'/'+view[0]+'_'+view[1]):
                        R,t=parse_transformation_file(join(args.dir_registration,args.substack_id,'_'.join(view)))
			va=-math.asin(R[2,0]);vb=math.atan2(R[2,1]/math.cos(va),R[2,2]/math.cos(va));vc=math.atan2(R[1,0]/math.cos(va),R[1,1]/math.cos(va))
			eul_angles=np.array([va,vb,vc])
			if np.linalg.norm(t)<args.norm_t and np.linalg.norm(eul_angles)<args.norm_r:
				is_good=True
		else:
			is_good=True   
   
		if is_good:
			marker_file=join(args.dir_findcells,args.substack_id,'_'.join(view),'ms.marker')
			if exists(marker_file):
				C= substack.load_markers(marker_file)
			else:
				C=[]
			valid_pairs[view]=C

	return valid_pairs


def get_visible_pairs(base_indir,substack_id,view_ids,lower_threshold=40.):
    visible_views={}
    num_vis=0
    for view_id in view_ids:
	substack = SubStack(base_indir+'/'+view_id,args.substack_id)
	patch = substack.get_volume()
	histogram = np.histogram(patch, bins=256,range=(0,256))[0]
	thresholds = threshold.multi_kapur(histogram, 2)
	if thresholds[1]>=lower_threshold:
	    visible_views[view_id]=1
	    num_vis+=1
	else:
	    visible_views[view_id]=0
    list_views=[]
    if num_vis>=2:
	if visible_views[view_ids[0]]==1 and visible_views[view_ids[1]]==1:
	    list_views.append(tuple((view_ids[0],view_ids[1])))
	if visible_views[view_ids[0]]==1 and visible_views[view_ids[3]]==1:
	    list_views.append(tuple((view_ids[0],view_ids[3])))
	if visible_views[view_ids[2]]==1 and visible_views[view_ids[1]]==1:
	    list_views.append(tuple((view_ids[2],view_ids[1])))
	if visible_views[view_ids[2]]==1 and visible_views[view_ids[3]]==1:
	    list_views.append(tuple((view_ids[2],view_ids[3])))
    if num_vis==1:
	if visible_views[view_ids[0]]==1:
	    list_views.append(tuple((view_ids[0],view_ids[1])))
	    list_views.append(tuple((view_ids[0],view_ids[3])))
	elif visible_views[view_ids[1]]==1:
	    list_views.append(tuple((view_ids[0],view_ids[1])))
	    list_views.append(tuple((view_ids[2],view_ids[1])))
	elif visible_views[view_ids[2]]==1:
	    list_views.append(tuple((view_ids[2],view_ids[1])))
	    list_views.append(tuple((view_ids[2],view_ids[3])))
	elif visible_views[view_ids[3]]==1:
	    list_views.append(tuple((view_ids[0],view_ids[3])))
	    list_views.append(tuple((view_ids[2],view_ids[3])))
    return list_views


def get_visible_pairs_from_dir(thresholds_dir, substack_id, view_ids, lower_threshold=40.):
    
    threshold_0=int(open(thresholds_dir+'/'+substack_id+'/'+view_ids[0]).readline().split(',')[1].rstrip())
    threshold_1=int(open(thresholds_dir+'/'+substack_id+'/'+view_ids[1]).readline().split(',')[1].rstrip())
    threshold_2=int(open(thresholds_dir+'/'+substack_id+'/'+view_ids[2]).readline().split(',')[1].rstrip())
    threshold_3=int(open(thresholds_dir+'/'+substack_id+'/'+view_ids[3]).readline().split(',')[1].rstrip())

    list_views=[]
    no_views=False
    two_views=False
    if threshold_0<lower_threshold and threshold_1<lower_threshold \
       and threshold_2<lower_threshold and threshold_3<lower_threshold:
            no_views=True

    if not no_views:
        if threshold_0>=lower_threshold and threshold_1>=lower_threshold:
                list_views.append(tuple(('000','090')))
                two_views=True     
        if threshold_0>=lower_threshold and threshold_3>=lower_threshold:
                list_views.append(tuple(('000','270')))        
                two_views=True     
        if threshold_2>=lower_threshold and threshold_1>=lower_threshold:
                list_views.append(tuple(('180','090')))        
                two_views=True     
        if threshold_2>=lower_threshold and threshold_3>=lower_threshold:
                list_views.append(tuple(('180','270')))        
                two_views=True     
        if not two_views:
            if threshold_0>=lower_threshold:
                    list_views.append(tuple(('000','090')))
                    list_views.append(tuple(('000','270')))
            if threshold_1>=lower_threshold:
                    list_views.append(tuple(('000','090')))
                    list_views.append(tuple(('180','090')))
            if threshold_2>=lower_threshold:
                    list_views.append(tuple(('180','090')))
                    list_views.append(tuple(('180','270')))
            if threshold_3>=lower_threshold:
                    list_views.append(tuple(('000','270')))
                    list_views.append(tuple(('180','270')))
                    
    return list_views



def merge_views(args):

    do_icp=args.do_icp
    view_ids=args.view_ids
    max_len=0
    hue=0.
    C_substack=[]

    if args.thresholds_dir is None:
        list_views = get_visible_pairs(args.indir,args.substack_id,view_ids)
    else:
        list_views = get_visible_pairs_from_dir(args.thresholds_dir, args.substack_id, view_ids)

    substack = SubStack(args.indir+'/'+view_ids[0],args.substack_id)
    valid_pairs = filter_valid_pairs(args,substack,list_views)

    hue_list=[0.,0.25,0.5,1.]
    i=0
    for k,v in valid_pairs.items():
            for c in v:
                    c.hue=hue_list[i]
            i+=1

    if len(valid_pairs.keys())==1:
            C_substack+=valid_pairs.values()[0]
    else:
            keys_view_1=[]
            keys_view_3=[]
            for view_key in valid_pairs.keys():
                    if view_key[0] == view_ids[2]:
                            keys_view_3.append(view_key)
                    elif view_key[0] == view_ids[0]:
                            keys_view_1.append(view_key)

    if len(valid_pairs)==2:
            if len(keys_view_1) == 2 or len(keys_view_3) == 2:
                    total_list=compute_fusion(substack,valid_pairs.values()[0],valid_pairs.values()[1],args.max_distance,match_distance=args.match_distance,do_icp=do_icp,verbose=args.verbose)
                    C_substack+=total_list
            elif len(keys_view_1) == 1 and len(keys_view_3) == 1:
                    C_view_3_t=transform_markers(args,substack,valid_pairs[keys_view_3[0]], view_ids)
                    total_list=compute_fusion(substack,valid_pairs[keys_view_1[0]],C_view_3_t,args.max_distance,match_distance=args.match_distance,do_icp=do_icp,verbose=args.verbose)
                    C_substack+=total_list
            else:
                    raise Exception('not valid list of views %s'%(str(valid_pairs)))

    elif len(valid_pairs)==3:
            if len(keys_view_1) == 2:
                    C_view_1=compute_fusion(substack,valid_pairs[keys_view_1[0]],valid_pairs[keys_view_1[1]],args.max_distance,match_distance=args.match_distance,do_icp=do_icp,verbose=args.verbose)
                    C_view_3_t=transform_markers(args,substack,valid_pairs[keys_view_3[0]], view_ids)
                    total_list=compute_fusion(substack,C_view_1,C_view_3_t,args.max_distance,match_distance=args.match_distance,do_icp=do_icp,verbose=args.verbose)
                    C_substack+=total_list
            elif len(keys_view_3) == 2:
                    C_view_3=compute_fusion(substack,valid_pairs[keys_view_3[0]],valid_pairs[keys_view_3[1]],args.max_distance,match_distance=args.match_distance,do_icp=do_icp,verbose=args.verbose)
                    C_view_3_t=transform_markers(args,substack,C_view_3, view_ids)
                    C_view_1=valid_pairs[keys_view_1[0]]
                    total_list=compute_fusion(substack,C_view_1,C_view_3_t,args.max_distance,match_distance=args.match_distance,do_icp=do_icp,verbose=args.verbose)
                    C_substack+=total_list
            else:
                    raise Exception('not valid list of views %s'%(str(valid_pairs)))

    elif len(valid_pairs)==4:
            if len(keys_view_1) == 2 and len(keys_view_3) == 2:
                    C_view_1=compute_fusion(substack,valid_pairs[keys_view_1[0]],valid_pairs[keys_view_1[1]],args.max_distance,match_distance=args.match_distance,do_icp=do_icp,verbose=args.verbose)
                    C_view_3=compute_fusion(substack,valid_pairs[keys_view_3[0]],valid_pairs[keys_view_3[1]],args.max_distance,match_distance=args.match_distance,do_icp=do_icp,verbose=args.verbose)
                    C_view_3_t=transform_markers(args,substack,C_view_3, view_ids)
                    total_list=compute_fusion(substack,C_view_1,C_view_3_t,args.max_distance,match_distance=args.match_distance,do_icp=do_icp,verbose=args.verbose)
                    C_substack+=total_list
            else:
                    raise Exception('not valid list of views %s'%(str(valid_pairs)))

    mkdir_p(args.outdir+'/'+args.substack_id)
    if len(C_substack)>0:
            C_substack = filter_outside_markers(C_substack,substack)
            substack.save_markers(args.outdir+'/'+args.substack_id+'/ms.marker', C_substack, floating_point=True)
    with open(args.outdir+'/'+args.substack_id+'/log','w') as f:
        f.write(args.substack_id+','+str(len(valid_pairs))+','+str(len(list_views))+'\n')

def get_parser():
	parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('indir', metavar='indir', type=str,
						help='needs indir/info.json, substacks, e.g. indir/10view_25, and GT files e.g. indir/10view_25-GT.marker')
	parser.add_argument('substack_id', metavar='substack_id', type=str,
						help='Substack identifier, e.g. 10view_25')
	parser.add_argument('dir_registration', metavar='dir_registration', type=str,
						help='dir of estimated rigid transformations')
	parser.add_argument('dir_findcells', metavar='dir_findcells', type=str,
						help="""where prediction results were saved, e.g. outdir/10view_25/ms.marker.
						As a special case, if outdir=="GT" then ground truth files are merged
						""")
	parser.add_argument('outdir', metavar='outdir', type=str,
						help='where to save the merged .marker file, e.g. merged.marker')
	parser.add_argument('--thresholds_dir', metavar='thresholds_dir', type=str,action='store',
						help='dir of thresholds, computed by a maximum entropy approach(Kapur et al.). The folder structure is thresholds_dir/substack_id/view_id')
	parser.add_argument('-i','--view_ids', nargs='+', help='list of view_ids es: 000 090 180 270', required=True)
	parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Verbose output.')
	parser.add_argument('--do_icp', dest='do_icp', action='store_true', help='do icp fusion')
	parser.add_argument('-m', '--match_distance', metavar='match_distance', dest='match_distance',
						action='store', type=float, default=4.0,
						help="""Max distance beyond which two markers are no longer matched in the ICP procedure""")
	parser.add_argument('-M', '--max_distance', metavar='max_distance', dest='max_distance',
						action='store', type=float, default=4.0,
						help="""Max distance beyond which two markers are no longer considered the same soma""")
	parser.add_argument('-r', metavar='norm_r', dest='norm_r',
						action='store', type=float, default=10e8,
						help="""Max euclidean norm of the vector composed by euler angles of the rotational component, beyond which the rigid transformation is considered failed""")
	parser.add_argument('-t', metavar='norm_t', dest='norm_t',
						action='store', type=float, default=10e8,
						help="""Max euclidean norm of the translational component, beyond which the rigid transformation is considered failed""")
	parser.set_defaults(verbose=False,do_icp=False)
	return parser


if __name__ == '__main__':
	parser = get_parser()
        args = parser.parse_args()
    	merge_views(args)

