# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import re


def from_xcl_to_txt(path_to_csv_original, path_to_csv_predict, output_txt_file_name, output_txt_file_name1, output_jsonlines_file_name, output_jsonlines_file_name_1):
    import csv
    print("parsing from CSV to txt- original...")
    # file for output
    g = open(output_txt_file_name, "w", encoding='utf-8')
    g_1 = open(output_txt_file_name1, "w", encoding='utf-8')

    with open(path_to_csv_original, newline='') as csvfile:
        text_row=''
        row_index=0;
        spam_reader = csv.reader(csvfile)
        for row in spam_reader:
            print(row)
            text_row=(', '.join(row))
            print(text_row)
            #text_row=text_row[text_row.find(',')+1:]
            # splitted_row=re.split('(\W)', text_row)
            # print(text_row)
            g.write(text_row)
            g.write("\n")
            row_index = row_index+1

        print("file is: ", output_txt_file_name)
    print("parsing from CSV to txt- ours...")

    with open(path_to_csv_predict, encoding='utf-8', newline='') as csvfile2:
        text_row_1=''
        row_index_1=0;
        spam_reader_1 = csv.reader(csvfile2)
        for row_1 in spam_reader_1:
            text_row_1=(', '.join(row_1))
            g_1.write(text_row_1)
            g_1.write("\n")
            row_index_1 = row_index_1+1
    print("file is: ", output_txt_file_name1)

    parse_file(output_txt_file_name, output_jsonlines_file_name_1, output_txt_file_name, output_txt_file_name1, output_jsonlines_file_name, output_jsonlines_file_name_1)


# def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
#     for e in self.evaluators:
#         e.update(predicted, gold, mention_to_predicted, mention_to_gold)



# def extract_mentions_to_predicted_clusters_from_clusters(gold_clusters):
#     mention_to_gold = {}
#     for gc in gold_clusters:
#         for mention in gc:
#             mention_to_gold[tuple(mention)] = gc
#     print(mention_to_gold)
#     return mention_to_gold



def parse_file(file_name_origin, file_name_ours, output_jsonlines_name_origin, output_jsonlines_name_ours , predict_jsonlines_name_origin, predict_jsonlines_name_ours):
    import xlsxwriter
    from metrics import CorefEvaluator, MentionEvaluator
    from utils import extract_clusters, extract_mentions_to_predicted_clusters_from_clusters, extract_clusters_for_decode

    # open files:
    path = r"C:\Users\AlmogReuveni\PycharmProjects\OursToJasonImp\input_for_parser"

    count_words_per_part = 0
    is_in_cluster = 0
    cur_cluster_num = ''
    para_num = 0
    count_words = 0


    ###
    t_start_index = -1
    t_tuple_index = tuple()
    t_tuple_list_per_cluster = []
    # a dictionary that maps clusters to their list of tuples:
    t_clus_to_list_dic = {'k': []}
    # list to hold lists of tuples for each para
    t_list_of_lists = []
    # list to hold all lists from all para
    t_list_of_lists_of_lists = []
    # list to keep first output:
    t_list_of_lists_of_lists_original = []
    # list to keep first output:
    t_list_of_lists_of_lists_ours = []
    ###

    # read line
    time = 0
    for time in {1, 2}:
        if time == 1:
            f_origin = open(r"C:\Users\AlmogReuveni\Development\s2e-coref\originalTxtFileFromCSV_testlimit", "r", encoding='utf-8')  # oldFormat
        elif time == 2:
            f_origin = open(r"C:\Users\AlmogReuveni\Development\s2e-coref\oursTxtFileFromCSV_testlimitMyFormat_beam3", "r",
                            encoding='utf-8')


        string_line = f_origin.readline()
        while (string_line!=''):
            print ("para num is: ", para_num)
            if para_num == 1:
                print ("here")
            # split to words by white spaces
            line_split = string_line.split()
            for word in line_split:
                count_words = count_words+ 1
                start_index = count_words_per_part
                if count_words_per_part == 3:
                    if time==2:
                        print(count_words_per_part)

                if is_in_cluster == 0:
                    # we are not in the middle of a cluster

                    if word[0] == '(':
                        # we started a new cluster
                        is_in_cluster = 1
                        t_start_index = count_words_per_part
                        # start of new string of indexes
                        # catch cluster number only:
                        if word[-1] == ')':
                            # singelton
                            cur_cluster_num = word[1:-1]
                            # create a tuple:
                            t_tuple_index = (start_index, start_index)
                            # find/create a matching cluster in the dict
                            if cur_cluster_num not in t_clus_to_list_dic:
                                t_clus_to_list_dic[cur_cluster_num] = []
                            # insert tuple to corresponding list from dict:
                            if(t_tuple_index[0]<800):
                                t_clus_to_list_dic[cur_cluster_num].append(t_tuple_index)
                            is_in_cluster = 0

                        else:
                            # no singleton, just beginning
                            cur_cluster_num = word[1:-1]
                            # create new key for cluster
                            if cur_cluster_num not in t_clus_to_list_dic:
                                t_clus_to_list_dic[cur_cluster_num] = []

                else:
                    # we are inside a cluster. check if end or middle
                    if word[-1] == ')':
                        # end of cluster

                        # create a tuple:
                        t_tuple_index = (t_start_index, count_words_per_part)
                        # no need to create cluster in dic- there must be one
                        # insert tuple to corresponding list from dict:
                        tup=()
                        if time == 1:
                            print (t_tuple_index)
                        if (t_tuple_index[0]<800):
                            t_clus_to_list_dic[cur_cluster_num].append(t_tuple_index)
                        t_start_index = -1
                        is_in_cluster = 0

                    else:
                        # middle of cluster
                        pass

                # to next word
                count_words_per_part = count_words_per_part+1

            count_words_per_part = 0
            is_in_cluster = 0
            string_line = f_origin.readline()
            para_num = para_num+1

            # convert dict to list:
            if (time==2):
                print ("ere")
            t_clus_to_list_dic.pop('k', None)
            tupi=()
            tupo=()
            for key in t_clus_to_list_dic:
                # print("time: ", time)
                # create tuple of tuples from list of tuples
                tupi = t_clus_to_list_dic[key]
                if len(tupi) < 1:
                    print(len(tupi))
                else:
                    a=tuple(t_clus_to_list_dic[key])
                    if len(tupi) == 1:
                        # only one tuple in list
                        b = tupi[0]
                        t_list_of_lists.append((tupi[0],()))
                    else:
                        t_list_of_lists.append(a)
                #else:
                 #   print("empty")
            t_list_of_lists_of_lists.append(t_list_of_lists)
            t_clus_to_list_dic = {'k': []}
            t_list_of_lists=[]
        f_origin.close()
        para_num = 0
        if time == 1:
            t_list_of_lists_of_lists_original = t_list_of_lists_of_lists.copy()
        elif time == 2:
            t_list_of_lists_of_lists_ours = t_list_of_lists_of_lists.copy()
        t_list_of_lists_of_lists.clear()

    my_count = 0
    write_count = 0
    #write to excel:
    workbook = xlsxwriter.Workbook('compare_eval_test_NoLimits_EladFormat_beam4.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write(0, 0, "original:")
    for item in t_list_of_lists_of_lists_original:
        write_count = write_count + 1
        worksheet.write(write_count, 0, str(item))

    write_count = 0
    worksheet.write(0, 1, "ours:")
    for item in t_list_of_lists_of_lists_ours:
        write_count = write_count + 1
        worksheet.write(write_count, 1, str(item))


    worksheet.write(0, 2, "f1 score:")
    for item in t_list_of_lists_of_lists_ours:
        coref_evaluator = CorefEvaluator()
        coref_evaluator.update(tuple(t_list_of_lists_of_lists_original[my_count]), tuple(t_list_of_lists_of_lists_ours[my_count]), extract_mentions_to_predicted_clusters_from_clusters(tuple(t_list_of_lists_of_lists_original[my_count])), extract_mentions_to_predicted_clusters_from_clusters(tuple(t_list_of_lists_of_lists_ours[my_count])))
        prec, rec, f1 = coref_evaluator.get_prf()
        worksheet.write(my_count+1, 2, str(f1))
        col=1
        for eval in coref_evaluator.evaluators:
            prec, rec, f1 = eval.get_prf()
            print("metric for: ", format(eval.name))

            print(f1)
            worksheet.write(0, 2+col, format(eval.name))
            worksheet.write(my_count+1, 2+col, str(f1))
            col = col + 1
        my_count = my_count+1
        if(my_count==625):
            break




    #print (original_opening_indexes)
    workbook.close()


    # for item in t_list_of_lists_of_lists:
    #     coref_evaluator = CorefEvaluator()
    #     # print(extract_mentions/_to_predicted_clusters_from_clusters(item))
    #     coref_evaluator.update(tuple(item), tuple(item), extract_mentions_to_predicted_clusters_from_clusters(item), extract_mentions_to_predicted_clusters_from_clusters(item))
    #     prec, rec, f1 = coref_evaluator.get_prf()
    #     print(f1)
    # return t_list_of_lists_of_lists


if __name__ == '__main__':
    # filePath=r'C:\Users\AlmogReuveni\Desktop\testOursToLines.txt'
    exl_path_original = r'C:\Users\AlmogReuveni\Desktop\testNoLimit\testNoLimitFix121Formated.csv'
    exl_path_ours = r'C:\Users\AlmogReuveni\Desktop\testNoLimit\pred_myformat_baseline50_450_beam4.csv'
    output_txt_file_name_original="originalTxtFileFromCSV_testlimit"
    output_txt_file_name_ours="oursTxtFileFromCSV_testlimitMyFormat_beam3"
    output_jsonlines_file_name_original="original_output_jsonlines_outcome_testNoLimit"
    output_jsonlines_file_name_ours="ours_output_jsonlines_outcome_testNoLimit"

    from_xcl_to_txt(exl_path_original, exl_path_ours, output_txt_file_name_original, output_txt_file_name_ours, output_jsonlines_file_name_original, output_jsonlines_file_name_ours)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
