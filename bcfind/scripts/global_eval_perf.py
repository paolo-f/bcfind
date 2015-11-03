__author__ = 'paciscopi'


import os,sys
import pandas as pd
import argparse

def main(args):

    eval_perf_folder = args.eval_perf_folder
    r=args.hi_local_max_radius
    b=args.mean_shift_bandwidth
    data_frame_markers = pd.read_csv(args.list_testset, dtype={'view1': str, 'view2': str, 'ss_id': str })
    list_substacks=[]
    for row in data_frame_markers.index:
        row_data=data_frame_markers.iloc[row]
	list_substacks.append((row_data['ss_id'],row_data['view1'],row_data['view2']))

    TP,FP,FN=0,0,0
    for ss in list_substacks:

        try:
		if args.multiview:
		    eval_log = open(args.eval_perf_folder.rstrip('//')+'/'+ ss[0]+'/'+ss[1]+'_'+ss[2]+'/eval.log', 'r')
		else:
		    eval_log = open(args.eval_perf_folder.rstrip('//')+'/'+ ss[0]+'/eval.log', 'r')

		title_items = eval_log.readline().split(',') #first line

		TP_index = [i for i, elem in enumerate(title_items) if elem=='TP'][0]
		FP_index = [i for i, elem in enumerate(title_items) if elem=='FP'][0]
		FN_index = [i for i, elem in enumerate(title_items) if elem=='FN'][0]

		res_items = eval_log.readline().split(',') #second lines

		TP += int(res_items[TP_index])
		FP += int(res_items[FP_index])
		FN += int(res_items[FN_index])
		eval_log.close()
        except IOError:
		print(ss,'file not existing')
		sys.exit(1)

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    F1 = 2*precision*recall/(precision+recall)
    print('P: %.2f , R: %.2f , F1: %.2f ==== TP: %d , FP: %d , FN: %d' % (precision*100,recall*100,F1*100,TP,FP,FN))
    final_f = open(args.eval_perf_folder.rstrip('//') + '/total_eval.log', 'w')
    final_f.write('%s,%s,%.2f,%.2f,%.2f,%d,%d,%d\n' % (r,b,precision*100,recall*100,F1*100,TP,FP,FN))
    final_f.close()


def get_parser():
    parser = argparse.ArgumentParser(description="""
    Final eval perf script
    """, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('eval_perf_folder', metavar='eval_perf_folder', type=str,
                        help='must contain the substacks with the eval.log file')
    parser.add_argument('list_testset', metavar='list_testset', type=str,
                        help='csv file of merged markers of the testset')
    parser.add_argument('-r', '--hi_local_max_radius', metavar='r', dest='hi_local_max_radius',
                        action='store', type=float, default=6,
                        help='Radius of the seed selection ball (r)')
    parser.add_argument('-R', '--mean_shift_bandwidth', metavar='R', dest='mean_shift_bandwidth',
                        action='store', type=float, default=5.5,
                        help='Radius of the mean shift kernel (R)')
    parser.add_argument('-m', '--multiview', dest='multiview', action='store_true',
                        help='Multiview mode')
    parser.set_defaults(multiview=False)
    return parser



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
