# ==============================================
# import
# ==============================================
import os
import config.system as sys_config

# ==============================================
# basepath where the original images of all subjects have been saved as numpy arrays 
# ==============================================
basepath = sys_config.project_data_freiburg_root #'/usr/bmicnas01/data-biwi-01/nkarani/students/nicolas/data/freiburg'

# ==============================================
# make a list of all subjects' relative paths
# This will contain 142 entries, although there are 126 subjects, as there are some subjects with multiple images.
# ==============================================
#all_subject_dirs = []
#_, dir_names_level0, _ = next(os.walk(basepath))
#
#for dir_name_this_subject_level0 in dir_names_level0:
#
#    _, dir_names_level1, _ = next(os.walk(os.path.join(basepath,
#                                                       dir_name_this_subject_level0)))
#
#    for dir_name_this_subject_level1 in dir_names_level1:    
#
#        _, dir_names_level2, _ = next(os.walk(os.path.join(basepath,
#                                                           dir_name_this_subject_level0,
#                                                           dir_name_this_subject_level1)))            
#
#        for dir_name_this_subject_level2 in dir_names_level2:  
#
#            subject_dir = os.path.join(dir_name_this_subject_level0,
#                                       dir_name_this_subject_level1,
#                                       dir_name_this_subject_level2)
#
#            all_subject_dirs.append(subject_dir)

# ==============================================
# Note that the order of the entries in all_subject_dirs depends on the behaviour of os.walk()
# I am not exactly certain how this order is defined.
# In order to ensure that the subject order does not change if the behaviour of os.walk() changes for some reason,
# I have run the above code once, printed the list all_subject_dirs and set it here again.
# ==============================================
all_subject_dirs = ['26/100180D1/100180D2',
 '71/100011BD/100011BE',
 '41/10016E22/10016E23',
 '7/10019B44/10019B45',
 '7/10000423/10000424',
 '104/10001BB0/10001BB1',
 '108/10008622/10008623',
 '110/100081C3/100081C4',
 '123/1000B3BB/1000B3BC',
 '144/100003AF/100003B0',
 '9/100005CE/100005CF',
 '9/100198C5/100198C6',
 '99/10000000/10000001',
 '124/1000B327/1000B328',
 '132/10000000/10000001',
 '85/10008E63/10008E64',
 '105/10001D28/10001D29',
 '105/1000870F/10008710',
 '67/1000337F/10003380',
 '14/100190C4/100190C5',
 '14/10002CA3/10002CA4',
 '81/100091B4/100091B5',
 '63/100039EF/100039F0',
 '142/10003DC5/10003DC6',
 '111/1000801E/1000801F',
 '78/10000000/10000001',
 '87/10008C57/10008C58',
 '12/1001927F/10019280',
 '12/10002A73/10002A74',
 '10/100196E0/100196E1',
 '10/10000718/10000719',
 '102/100002B8/100002B9',
 '21/1000331A/1000331B',
 '21/1001876F/10018770',
 '42/10016CF0/10016CF1',
 '35/10017524/10017525',
 '114/100068D6/100068D7',
 '83/10008F43/10008F44',
 '30/10017BB2/10017BB3',
 '45/10016940/10016941',
 '113/1000695C/1000695D',
 '3/10000000/10000001',
 '3/1001A34A/1001A34B',
 '29/10017D20/10017D21',
 '125/1000A3ED/1000A3EE',
 '34/100176D1/100176D2',
 '137/1000710C/1000710D',
 '139/10006F20/10006F21',
 '6/1000030E/1000030F',
 '6/10019D6B/10019D6C',
 '27/10017FC1/10017FC2',
 '11/1001942A/1001942B',
 '11/100028CE/100028CF',
 '72/10000F5E/10000F5F',
 '52/10016105/10016106',
 '44/10016A7F/10016A80',
 '120/100002BE/100002BF',
 '77/10000402/10000403',
 '62/10003BCB/10003BCC',
 '103/10000459/1000045A',
 '69/1000303B/1000303C',
 '128/100054EE/100054EF',
 '115/100035E1/100035E2',
 '97/10000173/10000174',
 '146/10000000/10000001',
 '94/10000718/10000719',
 '118/100034D1/100034D2',
 '59/10014A48/10014A49',
 '98/10000071/10000072',
 '122/1000D9AB/1000D9AC',
 '43/10016BD3/10016BD4',
 '148/100000B2/100000B3',
 '49/10016494/10016495',
 '86/10008D30/10008D31',
 '39/10017019/1001701A',
 '95/1000050D/1000050E',
 '76/10000675/10000676',
 '50/1001634B/1001634C',
 '101/10000142/10000143',
 '38/10017210/10017211',
 '130/100001C5/100001C6',
 '23/100184CD/100184CE',
 '31/10017A5B/10017A5C',
 '68/1000319C/1000319D',
 '143/100008CA/100008CB',
 '47/100166CF/100166D0',
 '66/100034EF/100034F0',
 '129/10000332/10000333',
 '109/10008485/10008486',
 '46/1001680B/1001680C',
 '70/100013A6/100013A7',
 '145/100041C1/100041C2',
 '135/1000AC5B/1000AC5C',
 '58/10014AE6/10014AE7',
 '24/100183C6/100183C7',
 '112/10006A59/10006A5A',
 '15/10018F01/10018F02',
 '15/10002E26/10002E27',
 '119/10000638/10000639',
 '73/10000D56/10000D57',
 '134/1000AD04/1000AD05',
 '64/10003813/10003814',
 '33/1001789E/1001789F',
 '22/1001860D/1001860E',
 '80/100092CE/100092CF',
 '90/100009DF/100009E0',
 '75/1000091D/1000091E',
 '74/10000AFC/10000AFD',
 '96/10000380/10000381',
 '17/10018918/10018919',
 '17/1000310C/1000310D',
 '127/10005671/10005672',
 '127/10005583/10005584',
 '4/1001A112/1001A113',
 '4/1000008F/10000090',
 '16/10018ADB/10018ADC',
 '16/10002FCA/10002FCB',
 '140/100044AC/100044AD',
 '37/100173EE/100173EF',
 '48/100165C7/100165C8',
 '25/100181C5/100181C6',
 '55/10015FEB/10015FEC',
 '147/10000268/10000269',
 '82/1000905A/1000905B',
 '100/10000000/10000001',
 '141/1000407F/10004080',
 '65/10003682/10003683',
 '117/1000353C/1000353D',
 '5/10019F69/10019F6A',
 '5/10000204/10000205',
 '28/10017ECA/10017ECB',
 '131/100000A4/100000A5',
 '133/1000DE15/1000DE16',
 '88/10008BD0/10008BD1',
 '136/1000ABC8/1000ABC9',
 '91/10000882/10000883',
 '57/10015E62/10015E63',
 '60/10003ECE/10003ECF',
 '121/10000000/10000001',
 '138/1000703B/1000703C',
 '149/10000000/10000001',
 '51/1001622E/1001622F']
            
# ==============================================
# Now, 'all_subject_dirs' has to be reordered, 
# such that the subjects segmented by Nicolas appear as the first 29 entries of the list.
# For this, we must first define the relative paths of the subjects that have been segmented by him.
# ==============================================
subject_dirs_segmented_by_Nicolas = ['14/100190C4/100190C5',
 '17/1000310C/1000310D',
 '148/100000B2/100000B3',
 '31/10017A5B/10017A5C',
 '3/10000000/10000001',
 '68/1000319C/1000319D',
 '114/100068D6/100068D7',
 '33/1001789E/1001789F',
 '64/10003813/10003814',
 '7/10000423/10000424',
 '146/10000000/10000001',
 '34/100176D1/100176D2',
 '77/10000402/10000403',
 '76/10000675/10000676',
 '102/100002B8/100002B9',
 '74/10000AFC/10000AFD',
 '66/100034EF/100034F0',
 '15/10018F01/10018F02',
 '120/100002BE/100002BF',
 '139/10006F20/10006F21',
 '24/100183C6/100183C7',
 '52/10016105/10016106',
 '35/10017524/10017525',
 '95/1000050D/1000050E',
 '65/10003682/10003683',
 '14/10002CA3/10002CA4',
 '94/10000718/10000719',
 '46/1001680B/1001680C',
 '30/10017BB2/10017BB3']
# ==============================================
# Although Nicolas also used os.walk() to define the order of subjects,
# he had saved the .npy arrays with a somewhat wierd naming convention.
# This lead os.walk() to read things in a different order as compared to what it read in all_subject_dirs.
# ==============================================

# ==============================================
# Now re-order all_subject_dirs
# such that the entries in subject_dirs_segmented_by_Nicolas appear at the top
# and all other entries are there subsequently in some random order.
# ==============================================
all_subject_dirs_reordered = subject_dirs_segmented_by_Nicolas
for subject_dir in all_subject_dirs:
    if subject_dir not in subject_dirs_segmented_by_Nicolas:
        all_subject_dirs_reordered.append(subject_dir)       
        
# ==============================================
# Now, the final subject order to be used for further processing is:
# ==============================================
SUBJECT_DIRS = all_subject_dirs_reordered
'''
printing SUBJECT_ORDER gives:
['14/100190C4/100190C5',
 '17/1000310C/1000310D',
 '148/100000B2/100000B3',
 '31/10017A5B/10017A5C',
 '3/10000000/10000001',
 '68/1000319C/1000319D',
 '114/100068D6/100068D7',
 '33/1001789E/1001789F',
 '64/10003813/10003814',
 '7/10000423/10000424',
 '146/10000000/10000001',
 '34/100176D1/100176D2',
 '77/10000402/10000403',
 '76/10000675/10000676',
 '102/100002B8/100002B9',
 '74/10000AFC/10000AFD',
 '66/100034EF/100034F0',
 '15/10018F01/10018F02',
 '120/100002BE/100002BF',
 '139/10006F20/10006F21',
 '24/100183C6/100183C7',
 '52/10016105/10016106',
 '35/10017524/10017525',
 '95/1000050D/1000050E',
 '65/10003682/10003683',
 '14/10002CA3/10002CA4',
 '94/10000718/10000719',
 '46/1001680B/1001680C',
 '30/10017BB2/10017BB3',
 '26/100180D1/100180D2',
 '71/100011BD/100011BE',
 '41/10016E22/10016E23',
 '7/10019B44/10019B45',
 '104/10001BB0/10001BB1',
 '108/10008622/10008623',
 '110/100081C3/100081C4',
 '123/1000B3BB/1000B3BC',
 '144/100003AF/100003B0',
 '9/100005CE/100005CF',
 '9/100198C5/100198C6',
 '99/10000000/10000001',
 '124/1000B327/1000B328',
 '132/10000000/10000001',
 '85/10008E63/10008E64',
 '105/10001D28/10001D29',
 '105/1000870F/10008710',
 '67/1000337F/10003380',
 '81/100091B4/100091B5',
 '63/100039EF/100039F0',
 '142/10003DC5/10003DC6',
 '111/1000801E/1000801F',
 '78/10000000/10000001',
 '87/10008C57/10008C58',
 '12/1001927F/10019280',
 '12/10002A73/10002A74',
 '10/100196E0/100196E1',
 '10/10000718/10000719',
 '21/1000331A/1000331B',
 '21/1001876F/10018770',
 '42/10016CF0/10016CF1',
 '83/10008F43/10008F44',
 '45/10016940/10016941',
 '113/1000695C/1000695D',
 '3/1001A34A/1001A34B',
 '29/10017D20/10017D21',
 '125/1000A3ED/1000A3EE',
 '137/1000710C/1000710D',
 '6/1000030E/1000030F',
 '6/10019D6B/10019D6C',
 '27/10017FC1/10017FC2',
 '11/1001942A/1001942B',
 '11/100028CE/100028CF',
 '72/10000F5E/10000F5F',
 '44/10016A7F/10016A80',
 '62/10003BCB/10003BCC',
 '103/10000459/1000045A',
 '69/1000303B/1000303C',
 '128/100054EE/100054EF',
 '115/100035E1/100035E2',
 '97/10000173/10000174',
 '118/100034D1/100034D2',
 '59/10014A48/10014A49',
 '98/10000071/10000072',
 '122/1000D9AB/1000D9AC',
 '43/10016BD3/10016BD4',
 '49/10016494/10016495',
 '86/10008D30/10008D31',
 '39/10017019/1001701A',
 '50/1001634B/1001634C',
 '101/10000142/10000143',
 '38/10017210/10017211',
 '130/100001C5/100001C6',
 '23/100184CD/100184CE',
 '143/100008CA/100008CB',
 '47/100166CF/100166D0',
 '129/10000332/10000333',
 '109/10008485/10008486',
 '70/100013A6/100013A7',
 '145/100041C1/100041C2',
 '135/1000AC5B/1000AC5C',
 '58/10014AE6/10014AE7',
 '112/10006A59/10006A5A',
 '15/10002E26/10002E27',
 '119/10000638/10000639',
 '73/10000D56/10000D57',
 '134/1000AD04/1000AD05',
 '22/1001860D/1001860E',
 '80/100092CE/100092CF',
 '90/100009DF/100009E0',
 '75/1000091D/1000091E',
 '96/10000380/10000381',
 '17/10018918/10018919',
 '127/10005671/10005672',
 '127/10005583/10005584',
 '4/1001A112/1001A113',
 '4/1000008F/10000090',
 '16/10018ADB/10018ADC',
 '16/10002FCA/10002FCB',
 '140/100044AC/100044AD',
 '37/100173EE/100173EF',
 '48/100165C7/100165C8',
 '25/100181C5/100181C6',
 '55/10015FEB/10015FEC',
 '147/10000268/10000269',
 '82/1000905A/1000905B',
 '100/10000000/10000001',
 '141/1000407F/10004080',
 '117/1000353C/1000353D',
 '5/10019F69/10019F6A',
 '5/10000204/10000205',
 '28/10017ECA/10017ECB',
 '131/100000A4/100000A5',
 '133/1000DE15/1000DE16',
 '88/10008BD0/10008BD1',
 '136/1000ABC8/1000ABC9',
 '91/10000882/10000883',
 '57/10015E62/10015E63',
 '60/10003ECE/10003ECF',
 '121/10000000/10000001',
 '138/1000703B/1000703C',
 '149/10000000/10000001',
 '51/1001622E/1001622F']
'''
