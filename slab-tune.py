import os
import sys

method= sys.argv[1]
slab_noise= float(sys.argv[2])
case= sys.argv[3]
total_seed= 3

base_script= 'python train.py --dataset slab --model_name slab --batch_size 128 --lr 0.1 --epochs 100 --out_classes 2 --train_domains 0.0 0.10 --test_domains 0.90 --slab_data_dim 2 ' + ' --slab_noise ' + str(slab_noise) + ' --n_runs ' + str(total_seed)

res_dir= 'results/slab/htune/slab_noise_' + str(slab_noise) + '/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

if method == 'mmd':
    base_script= base_script + ' --method_name mmd --gaussian 1 --conditional 0 '

elif method == 'c-mmd':
    base_script= base_script + ' --method_name mmd --gaussian 1 --conditional 1 '

elif method == 'coral':
    base_script= base_script + ' --method_name mmd --gaussian 0 --conditional 0 '
    
elif method == 'c-coral':    
    base_script= base_script + ' --method_name mmd --gaussian 0 --conditional 1 '

elif method == 'dann':
    base_script= base_script + ' --method_name dann --conditional 0 '

elif method == 'c-dann':
    base_script= base_script + ' --method_name dann --conditional 1 '

if method in ['dann', 'c-dann']:
    penalty_glist= [0.01, 0.1, 1.0, 10.0, 100.0]    
else:
    penalty_glist= [0.1, 1.0, 10.0]

grad_penalty_glist= [0.01, 0.1, 1.0, 10.0]
disc_steps_glist= [1, 2, 4, 8]

if case == 'train':
    if method in ['dann', 'c-dann']:    
        for penalty in penalty_glist:
            for grad_penalty in grad_penalty_glist:
                for disc_steps in disc_steps_glist:
                    script= base_script + ' --penalty_ws ' + str(penalty) + ' --grad_penalty ' + str(grad_penalty) + ' --d_steps_per_g_step ' + str(disc_steps) 
                    script= script + ' > ' + res_dir + str(method) + '_' + str(penalty) + '_' + str(grad_penalty) + '_' + str(disc_steps) + '.txt'
                    os.system(script)

    else:
        for penalty in penalty_glist:
            script= base_script + ' --penalty_ws ' + str(penalty) 
            script= script + ' > ' + res_dir + str(method) + '_' + str(penalty) + '.txt'
            os.system(script)

elif case == 'test':
    
    best_acc= -1
    best_err= -1
    best_case= ''
    
    if method in ['dann', 'c-dann']:    
        for penalty in penalty_glist:
            for grad_penalty in grad_penalty_glist:
                for disc_steps in disc_steps_glist:
                    f_name= res_dir + str(method) + '_' + str(penalty) + '_' + str(grad_penalty) + '_' + str(disc_steps) + '.txt'
                    f= open(f_name)
                    data= f.readlines()
                    # Source validation accuracy
                    mean= float(data[-4].replace('\n', '').split(' ')[-2])
                    err= float(data[-4].replace('\n', '').split(' ')[-1])
                    if mean > best_acc:
                        best_acc= mean
                        best_err= err
                        best_case= f_name
    else:
        for penalty in penalty_glist:
            f_name= res_dir + str(method) + '_' + str(penalty) + '.txt'
            f= open(f_name)
            data= f.readlines()
            # Source validation accuracy
            mean= float(data[-4].replace('\n', '').split(' ')[-2])
            err= float(data[-4].replace('\n', '').split(' ')[-1])
            if mean > best_acc:
                best_acc= mean
                best_err= err
                best_case= f_name
    
    print('Best Hparam for method: ', method, best_case)
    print('Best Accuracy', best_acc, best_err)