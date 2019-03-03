from random import randint
import os
from bs4 import BeautifulSoup
import json
from textblob import TextBlob
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
import datetime
from datetime import datetime,timedelta
import requests
import json
from stop_words import get_stop_words
import pymongo
import boto3
from scipy.spatial.distance import cosine
import goslate
import botocore
import psycopg2
import numpy as np
from operator import itemgetter
import pandas as pd
from flask import Flask, render_template, Response, request, redirect, url_for,session

application = app = Flask(__name__)

dbuser = os.environ['dbuser']
dbname = os.environ['dbname']
dbhost = os.environ['dbhost']
dbpassword= os.environ['dbpassword']
aws_access_key_id = os.environ['aws_access_key_id']
aws_secret_access_key = os.environ['aws_secret_access_key']
dbconnect = "dbname='"+dbname+"' user='"+dbuser+"' host='"+dbhost+"' password='"+dbpassword+"'"
app.secret_key = os.urandom(24)

def recent_arts(lang,days):
    col_name = lang +'_vecs'
    conn = psycopg2.connect("dbname='langalearn' user='ymroddi' host='lango84.cukbl7fyxfht.us-west-1.rds.amazonaws.com' password='royallord8413'")
    cur = conn.cursor()
    sql = "SELECT art_id,vec FROM " + col_name + " where dt > now() - interval '"+str(days)+ " days'"
    cur.execute(sql)
    recent_vecs  = cur.fetchall()
    conn.close()
    rec_vec_np = [[x[0],np.array(x[1])] for x in recent_vecs]

    return rec_vec_np

def user_prog_list(user_id):
    conn = psycopg2.connect("dbname='langalearn' user='ymroddi' host='lango84.cukbl7fyxfht.us-west-1.rds.amazonaws.com' password='royallord8413'")
    cur = conn.cursor()
    sql="SELECT lang,exer_type,exer_resp FROM exer_progress WHERE user_id = '" + user_id + "'"
    cur.execute(sql)
    exercises = cur.fetchall()
    conn.close()
    if len(exercises)>0:
        progrpt = progress_list(exercises)
        return progrpt
    else:
        return 'none'

def progress_list(exercise_list):
    progdf = pd.DataFrame(exercise_list)
    grped = progdf.groupby([0,1])
    grplist = [x for x in grped]
    prog_list = [[x[0],str(round((x[1][x[1][2]==True].count()[0]/x[1].count()[0])*100,1)),str(x[1].count()[0])] for x in grplist]
    task_list = []
    for x in prog_list:
        lang = x[0][0]
        if lang == 'de':
            langt = 'German'
        if lang == 'fr':
            langt = 'French'
        if lang == 'en':
            langt = 'English'
        if lang == 'es':
            langt = 'Spanish'
        exer = x[0][1]
        if exer == 'image':
            task = 'Image Identification'
        if exer == 'verb_comp':
            task = 'Verb Sentences'
        if exer == 'sent_comp':
            task = 'Sentence Completion'
        item = {'langt':langt,'task':task,'percent':x[1],'total':x[2]}
        task_list.append(item)
    return task_list

def friend_list(user_id_friend,status):
    conn = psycopg2.connect("dbname='langalearn' user='ymroddi' host='lango84.cukbl7fyxfht.us-west-1.rds.amazonaws.com' password='royallord8413'")
    cur = conn.cursor()
    sql="SELECT relationships.userid1,relationships.userid2,relationships.request_date,relationships.accept_date,user_ids.name FROM relationships,user_ids WHERE ((relationships.userid1 = "+str(user_id_friend)+" AND user_ids.id = "+str(user_id_friend)+") OR (relationships.userid2 = "+str(user_id_friend)+") AND user_ids.id = "+str(user_id_friend)+") AND relationships.status = " +str(status)
    cur.execute(sql)
    friend_rslt = cur.fetchall()
    conn.close()
    friends_list = []
    for x in friend_rslt:
        if x[0] != user_id_friend:
            friends_list.append(x)
        if x[1] != user_id_friend:
            friends_list.append(x)
    friends_list1 = [{'request_date':x[2].strftime('%m/%d/%Y'),'accept_date':x[3],'name':x[4]} for x in friends_list]
    return friends_list1

def fetch_recs_id(friend_ids):
    conn = psycopg2.connect("dbname='langalearn' user='ymroddi' host='lango84.cukbl7fyxfht.us-west-1.rds.amazonaws.com' password='royallord8413'")
    cur = conn.cursor()
    sql='SELECT id,name,native_lang,residence,login_status FROM user_ids WHERE id IN %s'
    cur.execute(sql,(friend_ids,))
    friend_list = cur.fetchall()
    dictrecs = [{'id':x[0],'name':x[1],'nativ_lang':x[2],'residence':x[3],'login_status':x[4]} for x in friend_list]
    return dictrecs

def cosine_rank(target_vec,time_vec,rec_num):
    dists = []
    for vec in time_vec:
        dist = 1 - cosine(target_vec,vec[1])
        item = [dist,vec[0]]
        dists.append(item)
    ranked = sorted(dists, key=itemgetter(0),reverse=True)
    return ranked[:rec_num]

def art_parser(link):
    r = requests.get(link)
    page = r.text
    soup = BeautifulSoup(page,"lxml")
    for x in soup('script'):
        x.decompose()
    for x in soup('link'):
        x.decompose()
    for x in soup('meta'):
        x.decompose()
    title = soup.title.string
    paras = soup('p')
    atriclestrip = [art.get_text() for art in paras]
    art = ' '.join(atriclestrip)
    return art,link,title


def load_models_s3(lang):
    bucket_name = 'langlearn84'
    KEY1 = lang + 'model3.model'
    KEY2 = lang + 'model3.model.trainables.syn1neg.npy'
    KEY3 = lang + 'model3.model.wv.vectors.npy'
    # if d2v model not in directory, download it
    if not os.path.exists(KEY1):

        s3 = boto3.resource(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
        )

        try:
            s3.Bucket(bucket_name).download_file(KEY1, KEY1)
            print(KEY1)
            s3.Bucket(bucket_name).download_file(KEY2, KEY2)
            print(KEY2)
            s3.Bucket(bucket_name).download_file(KEY3, KEY3)
            print(KEY3)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise
        lang_model = Doc2Vec.load(KEY1)
    else:
        lang_model = Doc2Vec.load(KEY1)
    return lang_model

def langname(lang_select):
    if lang_select == 'es':
        langt = 'Spanish'
    if lang_select == 'fr':
        langt = 'French'
    if lang_select == 'de':
        langt = 'German'
    if lang_select == 'en':
        langt = 'English'
    return langt

def list_routes():
    return ['%s' % rule for rule in app.url_map.iter_rules()]

@application.route("/")
def hello():
    routelinks = list_routes()
    html = "<h1 style='color:blue'>Routes</h1>"
    for link in routelinks:
        html += '<P><H3>'+link+'</H3></P>'
    return html

@application.route("/apis/single_art", methods=['POST'])
def single_art():
    #trans_art = request.json['trans_art']
    trans_lang = request.json['trans_lang']
    art_id = request.json['art_id']
    colnm = trans_lang + '_arts'
    conn = psycopg2.connect(dbconnect)
    cur = conn.cursor()
    sql = "SELECT link,title,article,art_id FROM " + colnm + " WHERE art_id = '" + art_id + "'"
    cur.execute(sql)
    article1  = cur.fetchone()
    dictart = {'link':article1[0],'title':article1[1],'article':article1[2],'art_id':article1[3]}
    conn.close()
    resp=json.dumps(dictart)
    return resp

@application.route("/apis/link_search", methods=['POST'])
def link_search():
    def trans_art(link,trans_lang):
        art,link,title = art_parser(link)
        trans_art = [str(TextBlob(art).translate(to=trans_lang))]
        return trans_art,title
    #trans_art = request.json['trans_art']
    trans_lang = request.json['trans_lang']
    link = request.json['link']
    daterange = request.json['daterange']

    trans_art,title = trans_art(link,trans_lang)

    if trans_lang == 'es':
        langt = 'Spanish'
        lang_model = eslang_model
        colnm = trans_lang +'_arts'
    if trans_lang == 'fr':
        langt = 'French'
        lang_model = frlang_model
        colnm = trans_lang +'_arts'
    if trans_lang == 'de':
        langt = 'German'
        lang_model = delang_model
        colnm = trans_lang +'_arts'

    stop_words = get_stop_words(trans_lang)
    histnostop = [[i for i in doc.lower().split() if i not in stop_words] for doc in trans_art]
    dlhist_tagged = [TaggedDocument(doc,[i]) for i,doc in enumerate(histnostop)]
    ## infer vectors from current doc2model
    trans_lang_vec = [lang_model.infer_vector(doc.words) for doc in dlhist_tagged]
    rec_num = 20
    #sims = lang_model.docvecs.most_similar(trans_lang_vec, topn=rec_num)
    #load to time matrix
    vec_range = recent_arts(trans_lang,daterange)
    rankedvec = cosine_rank(trans_lang_vec,vec_range,rec_num)

    sims1= [x[1] for x in rankedvec]
    sims2= tuple(sims1)

    conn = psycopg2.connect(dbconnect)
    cur = conn.cursor()
    sql="SELECT link,title,art_id FROM " + colnm + " WHERE art_id IN %s"
    cur.execute(sql,(sims2,))
    recs = cur.fetchall()
    dictrecs = [{'link':x[0],'title':x[1],'art_id':x[2]} for x in recs]
    conn.close()
    payload = {'recs':dictrecs,'link':link,'title':title,'trans_lang':trans_lang,'langt':langt}
    resp=json.dumps(payload)
    return resp

@application.route("/apis/link_search_pg", methods=['POST'])
def link_search_pg():
    def trans_art(link,trans_lang):
        art,link,title = art_parser(link)
        trans_art = [str(TextBlob(art).translate(to=trans_lang))]
        return trans_art,title
    #trans_art = request.json['trans_art']
    trans_lang = request.json['trans_lang']
    link = request.json['link']

    trans_art,title = trans_art(link,trans_lang)

    if trans_lang == 'es':
        langt = 'Spanish'
        lang_model = eslang_model
        colnm = trans_lang +'_arts'
    if trans_lang == 'fr':
        langt = 'French'
        lang_model = frlang_model
        colnm = trans_lang +'_arts'
    if trans_lang == 'de':
        langt = 'German'
        lang_model = delang_model
        colnm = trans_lang +'_arts'

    stop_words = get_stop_words(trans_lang)
    histnostop = [[i for i in doc.lower().split() if i not in stop_words] for doc in trans_art]
    dlhist_tagged = [TaggedDocument(doc,[i]) for i,doc in enumerate(histnostop)]
    ## infer vectors from current doc2model
    trans_lang_vec = [lang_model.infer_vector(doc.words) for doc in dlhist_tagged]
    rec_num = 20
    sims = lang_model.docvecs.most_similar(trans_lang_vec, topn=rec_num)
    sims1= [int(x[0]) for x in sims]
    sims2= tuple(sims1)

    conn = psycopg2.connect(dbconnect)
    cur = conn.cursor()
    sql="SELECT link,title,art_id FROM " + colnm + " WHERE id IN %s"
    cur.execute(sql,(sims2,))
    recs = cur.fetchall()
    dictrecs = [{'link':x[0],'title':x[1],'art_id':x[2]} for x in recs]
    conn.close()
    payload = {'recs':dictrecs,'link':link,'title':title,'trans_lang':trans_lang,'langt':langt}
    dump = [payload,sims2]
    resp=json.dumps(payload)
    return resp

@application.route("/apis/vocab_ins", methods=['POST'])
def vocab_ins():
    vocab_word = request.json['vocab_word']
    trans_word = request.json['trans_word']
    user_id = request.json['user_id']
    date = request.json['date']
    conn = psycopg2.connect("dbname='langalearn' user='ymroddi' host='lango84.cukbl7fyxfht.us-west-1.rds.amazonaws.com' password='royallord8413'")
    cur = conn.cursor()
    QueryData = "('"+ user_id  +"','" + vocab_word  + "','" + trans_word  +"','"+ date  +"')"
    cur.execute('INSERT INTO vocab (user_id,word,translation,date) VALUES ' + QueryData)
    conn.commit()
    conn.close
    payload = { 'vocab_word': vocab_word, 'trans_word': trans_word}
    resp=json.dumps(payload)
    return resp

@application.route("/apis/exer_progress", methods=['POST'])
def exer_progress():
    lang_select = request.json['lang_select']
    item = request.json['item']
    user_id = request.json['user_id']
    exer_date = request.json['exer_date']
    exer_type = request.json['exer_type']
    exer_resp = request.json['exer_resp']

    conn = psycopg2.connect("dbname='langalearn' user='ymroddi' host='lango84.cukbl7fyxfht.us-west-1.rds.amazonaws.com' password='royallord8413'")
    cur = conn.cursor()
    QueryData = "('"+ lang_select  +"','" + item  + "','" + user_id  +"','"+ exer_date +"','"+ exer_type +"','"+ str(exer_resp) +"')"
    cur.execute('INSERT INTO exer_progress (lang,item,user_id,exer_date,exer_type,exer_resp) VALUES ' + QueryData)
    conn.commit()
    conn.close
    payload = { 'item ': item , 'exer_resp': exer_resp}
    resp=json.dumps(payload)
    return resp



@app.route("/apis/art_recs", methods=['GET','POST'])
def art_recs():
    lang_select = request.args.get('values')
    trans_lang = request.args.get('trans_lang')
    user_id = session.get('user_id')
    db_name = 'arts_recs'
    colnm = trans_lang +'_arts'
    link_recs = []

    clusters = [['46c47616895140a28fcf5f7c368357ae',
         '43db6fcc5bd14b4584d78478ef8a4831',
         '39ff78c46b1b4db6baa2a84a670c84ba'],
          ['6404d798aa1547fca35f11693328d318',
         '424be85fad2c4448b944e7e795df857e',
         '008a5bdb929a4360b2a113feed312bf5'],
          ['1bd11f965c934560b0caa0c7e29388d1',
         '213478cc4a904f279ef38e52d2b0e7d4',
         'bb77defbe39c4d0da78ca28c9d82a8bd']
         ]
    rec_clusters = []
    for cluster in clusters:
        conn = psycopg2.connect(dbconnect)
        cur = conn.cursor()
        sql="SELECT link,title,art_id FROM " + colnm + " WHERE art_id IN %s"
        cur.execute(sql,(cluster,))
        recs = cur.fetchall()
        dictrecs = [{'link':x[0],'title':x[1],'art_id':x[2]} for x in recs]
        rec_clusters.append(dictrecs)
        conn.close()
    #link_recs = [[gcol.find_one({ "id" : db_id },projection={'_id': False,'title':True,'link':True,'id':True}) for db_id in db_ids] for db_ids in recs]

    return rec_clusters

@app.route("/apis/image_rec", methods=['GET','POST'])
def image_rec():
    lang_select = request.json['lang_select']
    colnm  = lang_select+'_pics'

    langt = langname(lang_select)

    conn = psycopg2.connect(dbconnect)
    cur = conn.cursor()
    sql="SELECT link,term FROM " + colnm + " ORDER BY random() LIMIT 1"
    cur.execute(sql)
    pic= cur.fetchall()
    conn.close()
    payload = {'link':pic[0][0],'term':pic[0][1],'lang_select':lang_select,'langt':langt}
    resp=json.dumps(payload)
    return resp

@app.route("/apis/verbcompletion", methods=['GET','POST'])
def verb_comp():
    lang_select = request.json['lang_select']
    native_lang = request.json['native_lang']
    db_name = 'sent_combo'

    def verb_random():
        conn = psycopg2.connect(dbconnect)
        cur = conn.cursor()
        sql="SELECT verb FROM sc_verbs ORDER BY random() LIMIT 1"
        cur.execute(sql)
        verb = cur.fetchone()
        conn.close()
        return verb[0]

    def noun_random():
        conn = psycopg2.connect(dbconnect)
        cur = conn.cursor()
        sql="SELECT noun FROM sc_nouns ORDER BY random() LIMIT 1"
        cur.execute(sql)
        noun = cur.fetchone()
        conn.close()
        return noun[0]

    def gen_sent():
        verb = verb_random()
        noun = noun_random()
        article = ['a','the']
        j = randint(0,1)
        art = article[j]
        return verb + ' ' + art + ' ' + noun
    sent =  gen_sent()
    blob = TextBlob(sent)
    learn_sent = blob.translate(to=lang_select)
    native_sent = str(learn_sent.translate(to=native_lang)).capitalize()
    trans_sent = str(learn_sent).capitalize()

    langt = langname(lang_select)

    payload = {'trans_sent':trans_sent,'native_sent':native_sent,'lang_select':lang_select,'langt':langt}
    resp=json.dumps(payload)
    return resp

@app.route("/apis/sentcompletion", methods=['GET','POST'])
def sent_comp():
    lang_select = request.json['lang_select']
    pos1 = request.json['pos']
    colnm  = lang_select+'_sents'
    conn = psycopg2.connect(dbconnect)
    cur = conn.cursor()
    sql="SELECT blanks,answer,speech,id FROM " + colnm + " WHERE pos = '" + pos1  + "' ORDER BY random() LIMIT 1"
    cur.execute(sql)
    sent = cur.fetchall()
    conn.close()

    langt = langname(lang_select)

    payload = {'item_id':str(sent[0][3]),'exer_blanks':sent[0][0],'translate':sent[0][2],'answer':sent[0][1],'lang_select':lang_select,'langt':langt}
    resp=json.dumps(payload)
    return resp

@app.route("/apis/translate_tt", methods=['GET','POST'])
def translate_tt():
    lang = request.json['lang']
    text = request.json['text']
    gs = goslate.Goslate()
    translatedtext = gs.translate(text,lang)
    payload = {'translatedText':translatedtext}
    resp=json.dumps(payload)
    return resp

@app.route("/apis/prog_list", methods=['GET','POST'])
def prog_list():
    user_id = request.json['user_id']
    progressuserid = user_prog_list(user_id)
    payload = {'progressuserid':progressuserid}
    resp=json.dumps(payload)
    return resp

@app.route("/apis/user_detail", methods=['GET','POST'])
def user_detail():
    user_id = request.json['user_id']
    login_status = 'on_line'
    last_login = str(datetime.now())
    conn = psycopg2.connect("dbname='langalearn' user='ymroddi' host='lango84.cukbl7fyxfht.us-west-1.rds.amazonaws.com' password='royallord8413'")
    cur = conn.cursor()
    sql = "UPDATE user_ids SET login_status = '" + login_status +"',last_login='"+ last_login+"' WHERE user_id = '" + user_id + "'"
    cur.execute(sql)
    conn.commit()
    sql1 = "select id,native_lang,learning,user_id,name from user_ids where user_id='"+ user_id + "'"
    cur.execute(sql1)
    user_rslt = cur.fetchone()
    native_lang = user_rslt[1]
    user_pk_id = user_rslt[0]
    learning = user_rslt[2]
    user_id = user_rslt[3]
    name = user_rslt[4]
    conn.close()
    payload = {'native_lang':native_lang,'learning':learning,'user_id':user_id,'name':name}
    resp=json.dumps(payload)
    return resp

@app.route("/apis/friends_search", methods=['GET','POST'])
def friends_search():
    age_src = request.json['age_src']
    srch_native_lang = request.json['srch_native_lang']
    gender = request.json['gender']
    conn = psycopg2.connect("dbname='langalearn' user='ymroddi' host='lango84.cukbl7fyxfht.us-west-1.rds.amazonaws.com' password='royallord8413'")
    cur = conn.cursor()
    if len(age_src)>0:
        agelow = str(age_src[0])
        agehigh = str(age_src[1])
        age_qry = " AND age BETWEEN " + agelow + " AND " + agehigh
    else:
        age_qry = ''
    sql="SELECT name,native_lang,sex,residence,age,(now() - last_login),id FROM user_ids WHERE sex = '" + gender + "' AND native_lang = '" + srch_native_lang + "'" + age_qry
    cur.execute(sql)
    friend_rslt = cur.fetchall()
    conn.close()
    friends = [{'name':item[0],'native_lang':item[1],'gender':item[2],'residnce':item[3],'age':str(item[4]),"last_login_time":str(item[5].days) + ' days','id':item[6]} for item in friend_rslt]
    payload = {'friends':friends}
    resp=json.dumps(payload)
    return resp

@app.route("/apis/friends_relationship", methods=['GET','POST'])
def friends_relationship():
    user_id_friend = request.json['user_id_friend']
    status = request.json['status']
    user_friends = friend_list(user_id_friend,status)
    payload = user_friends
    resp=json.dumps(payload)
    return resp

@app.route("/apis/friend_request", methods=['GET','POST'])
def friends_request():
    user_id_friend = request.json['user_id_friend']
    req_type = request.json['req_type']
    requested_id = request.json['requested_id']
    conn = psycopg2.connect("dbname='langalearn' user='ymroddi' host='lango84.cukbl7fyxfht.us-west-1.rds.amazonaws.com' password='royallord8413'")
    cur = conn.cursor()
    if req_type == 'friend_request':
        request_date  = datetime.now()
        status = 1
        req_data = (user_id_friend,requested_id,status,request_date)
        sql='INSERT INTO relationships (userid1,userid2,status,request_date) VALUES (%s, %s, %s,%s)'
        message = "Request Made " + request_date.strftime('%m/%d/%Y')
        cur.execute(sql,req_data )
        conn.commit()
        conn.close()
    if req_type == 'friend_acceptance':
        status = str(2)
        accept_date  = datetime.now()
        accept_data = (status,accept_date)
        sql='UPDATE relationships (status,accept_date) VALUES (%s, %s)'
        message = "Request Accept " + accept_date.strftime('%m/%d/%Y')
        cur.execute(sql,accept_data)
        conn.commit()
        conn.close()

    payload = {'message':message}
    resp=json.dumps(payload)

    return resp

if __name__ == '__main__':

    app.debug = True
    application.run(host='0.0.0.0',port='8484')
