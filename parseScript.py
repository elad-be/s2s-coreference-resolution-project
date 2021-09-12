# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import re


def from_xcl_to_txt(path_to_csv, output_txt_file_name1, output_jsonlines_file_name1):
    import csv
    # file for output
    g = open(output_txt_file_name1, "w", encoding='utf-8')

    with open(path_to_csv, newline='') as csvfile:
        text_row=''
        row_index=0;
        spam_reader = csv.reader(csvfile)
        for row in spam_reader:
            text_row=(', '.join(row))
            #text_row=text_row[text_row.find(',')+1:]
            # splitted_row=re.split('(\W)', text_row)
            # print(text_row)
            g.write(text_row)
            g.write("\n")
            row_index = row_index+1

    parse_file(output_txt_file_name1, output_jsonlines_file_name1)


def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
    for e in self.evaluators:
        e.update(predicted, gold, mention_to_predicted, mention_to_gold)


def extract_mentions_to_predicted_clusters_from_clusters(gold_clusters):
    mention_to_gold = {}
    for gc in gold_clusters:
        for mention in gc:
            mention_to_gold[tuple(mention)] = gc
    print(mention_to_gold)
    return mention_to_gold



def parse_file(file_name, output_jsonlines_name):
    import re
    # open files:
    path=r"C:\Users\AlmogReuveni\PycharmProjects\OursToJasonImp\input_for_parser"
    g = open(output_jsonlines_name, "w", encoding='utf-8')
    h = open ("printing_outcome.txt", "w", encoding='utf-8')

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
    t_list_of_lists_of_lists =[]
    ###

    h.write("start")
    # read line
    f = open(r"C:\Users\AlmogReuveni\PycharmProjects\OursToJasonImp\input_for_parser1", "r", encoding='utf-8')  # oldFormat
    string_line = f.readline()
    while (string_line!=''):
        print ("para num is: ", para_num)
        # split to words by white spaces
        line_split = string_line.split()
        for word in line_split:
            count_words = count_words+ 1
            start_index = count_words_per_part

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
                        t_clus_to_list_dic[cur_cluster_num].append(t_tuple_index)
                        is_in_cluster = 0

                    else:
                        # no singleton, just beginning
                        cur_cluster_num = word[1:]
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
        string_line = f.readline()
        para_num = para_num+1

        # convert dict to list:
        t_clus_to_list_dic.pop('k', None)
        for key in t_clus_to_list_dic:
            t_list_of_lists.append(t_clus_to_list_dic[key])
        t_list_of_lists_of_lists.append(t_list_of_lists)
        t_clus_to_list_dic = {'k': []}
        t_list_of_lists=[]

    f.close()
    g.close()
    h.close()

    for item in t_list_of_lists_of_lists:
        extract_mentions_to_predicted_clusters_from_clusters(item)
    return t_list_of_lists_of_lists


if __name__ == '__main__':
    # filePath=r'C:\Users\AlmogReuveni\Desktop\testOursToLines.txt'
    exl_path_to_read=r'C:\Users\AlmogReuveni\Desktop\znlp1908\devWithTextNoWordsLimit.csv'
    output_txt_file_name="input_for_parser"
    output_jsonlines_file_name="output_jsonlines_outcome"
    from_xcl_to_txt(exl_path_to_read, output_txt_file_name,output_jsonlines_file_name)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
