import xml.etree.ElementTree as ET
import csv
import os
from clean_data import clean_data
from tqdm import tqdm
cur_dir = os.getcwd()


def read(filename, output_name, clean_tag):
    #record by list
    output = []
    output_count = 0
    for data in filename:
        #use xml package to fetch data
        tree = ET.parse(os.path.join(cur_dir,data))
        root = tree.getroot()
        for att in tqdm(root, ascii = True):
            #fetch Thread
            child_ques = att.find("Thread");
            #fetch RelQuestion
            child_ques_att = child_ques.find("RelQuestion")

            for ans_att in child_ques.findall("RelComment"):
                #fetch OrgQBody , RelQBody, OrgQSubject, RelQSubject

                OrgQBody = att.find("OrgQBody").text
                RelQBody = child_ques_att.find("RelQBody").text
                OrgQSubject = att.find("OrgQSubject").text
                RelQSubject = child_ques_att.find("RelQSubject").text

                #clean_tag (nlp preprocessing) need time, use if-else to eliminate consuming time
                if (clean_tag):
                    if output_count == 0:
                        #nlp preprocessing
                        OrgQBody = clean_data.clean(OrgQBody)
                        RelQBody = clean_data.clean(RelQBody)
                        OrgQSubject =  clean_data.clean(OrgQSubject)
                        RelQSubject = clean_data.clean(RelQSubject)
                    elif (att.find("Thread").attrib["THREAD_SEQUENCE"] == output[output_count - 1][3]):
                        #repeating items
                        OrgQBody = output[output_count - 1][2]
                        RelQBody = output[output_count - 1][5]
                        OrgQSubject = output[output_count - 1][1]
                        RelQSubject = output[output_count - 1][4]
                    else:
                        #nlp preprocessing

                        OrgQBody = clean_data.clean(OrgQBody)
                        RelQBody = clean_data.clean(RelQBody)
                        OrgQSubject =  clean_data.clean(OrgQSubject)
                        RelQSubject = clean_data.clean(RelQSubject)

                ans_sheet = [att.attrib["ORGQ_ID"], OrgQSubject, OrgQBody, att.find("Thread").attrib["THREAD_SEQUENCE"], RelQSubject, RelQBody, child_ques_att.attrib["RELQ_RELEVANCE2ORGQ"]]
                #fetch RELC_ID , RELC_RELEVANCE2RELQ, RelCText
                ans_sheet.append(ans_att.attrib["RELC_ID"])
                ans_sheet.append(ans_att.attrib["RELC_RELEVANCE2RELQ"])

                RelCText = ans_att.find("RelCText").text
                #nlp preprocessing

                if  (clean_tag):
                    RelCText = clean_data.clean(RelCText)

                ans_sheet.append(RelCText)

                output.append(ans_sheet)
                output_count = output_count + 1

    #save file
    title = ["ORGQ_ID", "OrgQSubject", "OrgQBody","THREAD_SEQUENCE", "RelQSubject", "RelQBody", "RELQ_RELEVANCE2ORGQ","RELC_ID", "RELC_RELEVANCE2RELQ", "RelCText"]
    with open(os.path.join(cur_dir,output_name), 'w', newline='', encoding="utf-8") as myfile:
        wr = csv.writer(myfile)
        wr.writerow(title)
        for i in output:
            wr.writerow(i)

if __name__ == '__main__':
    #only use test data
    test = ["SemEval2017-task3-English-test-input.xml"]
    clean_tag = 1
    read(test, "test_use.csv", clean_tag)
