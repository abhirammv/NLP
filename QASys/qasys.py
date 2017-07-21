#Implement baseline for all files first and get the data from both .dep and .par files
#Then implement the usage of information from constituency parses
#Implement answer type detection
#Add the argument parser to parse the process_stories.txt file
#Need to add wordnet functionality


import nltk, re
import csv
from collections import OrderedDict
from nltk.tree import Tree
from collections import defaultdict
from nltk.corpus import wordnet as wn

#Reads the file
def read_file(filename):
    fh = open(filename, 'r')
    text = fh.read()
    fh.close()
    return text

#Reads the .par
def read_con_parses(parfile, type):
    if type == "story":
        parfile = parfile + ".story.par"
    elif type == "sch":
        parfile = parfile + ".sch.par"

    fh = open(parfile, 'r')
    lines = fh.readlines()
    fh.close()
    return [Tree.fromstring(line) for line in lines]

#Gets the questions
def getQA(filename):
    content = open(filename, 'rU', encoding='latin1').read()
    question_dict = {}
    for m in re.finditer(r"QuestionID:\s*(?P<id>.*)\nQuestion:\s*(?P<ques>.*)\nDifficulty:\s*(?P<diff>.*)\nType:\s*(?P<type>.*)\n*", content):
        qid = m.group("id")
        question_dict[qid] = {}
        question_dict[qid]['Question'] = m.group("ques")
        question_dict[qid]['Difficulty'] = m.group("diff")
        question_dict[qid]['Type'] = m.group("type")
    return question_dict

def get_data_dict(fname):
    data_dict = {}
    data_types = ["story", "sch", "questions"]
    parser_types = ["par", "dep"]
    for dt in data_types:
        data_dict[dt] = read_file(fname + "." + dt)
        for tp in parser_types:
            data_dict['{}.{}'.format(dt, tp)] = read_file(fname + "." + dt + "." + tp)
    return data_dict

#Gets the bag of words
def get_bow(tagged_tokens, stopwords):
    return set([t[0].lower() for t in tagged_tokens if t[0].lower() not in stopwords])

def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences

#Gets the baseline answers
def get_candidate_answers(question, text):
    stopwords = set(nltk.corpus.stopwords.words("english"))

    # Collect all the candidate answers
    candidate_answers = []
    qbow = get_bow(question, stopwords)
    sentences = nltk.sent_tokenize(text)
    for i in range(0, len(sentences)):
        sent = sentences[i]
        # A list of all the word tokens in the sentence
        sbow = get_bow(sent, stopwords)

        # Count the # of overlapping words between the Q and the A
        # & is the set intersection operator
        overlap = len(qbow & sbow)

        candidate_answers.append((overlap, i, sent))

        # Sort the results by the first element of the tuple (i.e., the count)
        # Sort answers from smallest to largest by default, so reverse it
        # Make sure to check about whether the results are null.

    return sorted(candidate_answers, key=lambda x: x[0], reverse=True)

def pattern_matcher(pattern, tree):
    for subtree in tree.subtrees():
        node = matches(pattern, subtree)
        if node is not None:
            return node
    return None


def matches(pattern, root):
    # Base cases to exit our recursion
    # If both nodes are null we've matched everything so far
    if root is None and pattern is None:
        return root

    # We've matched everything in the pattern we're supposed to (we can ignore the extra
    # nodes in the main tree for now)
    elif pattern is None:
        return root

    # We still have something in our pattern, but there's nothing to match in the tree
    elif root is None:
        return None

    # A node in a tree can either be a string (if it is a leaf) or node
    plabel = pattern if isinstance(pattern, str) else pattern.label()
    rlabel = root if isinstance(root, str) else root.label()

    # If our pattern label is the * then match no matter what
    if plabel == "*":
        return root
    # Otherwise they labels need to match
    elif plabel == rlabel:
        # If there is a match we need to check that all the children match
        # Minor bug (what happens if the pattern has more children than the tree)
        for pchild, rchild in zip(pattern, root):
            match = matches(pchild, rchild)
            if match is None:
                return None
        return root

    return None

#Gets the answers by matching the .par
def get_con_answers(question, par_file_data, sent_index):
    wh_word = nltk.word_tokenize(question)[0]
    if wh_word == "What":

        try:
            pattern = nltk.ParentedTree.fromstring("(VP (*))")
            tree = par_file_data[sent_index]
            subtree = pattern_matcher(pattern, tree)
            answer = " ".join(subtree.leaves())
            return str(answer)
        except:
            pattern = nltk.ParentedTree.fromstring("(NP (*))")
            #pattern = nltk.ParentedTree.fromstring("(S)")
            tree = par_file_data[sent_index]
            subtree = pattern_matcher(pattern, tree)
            answer = " ".join(subtree.leaves())
            return str(answer)

    elif wh_word == "Who":

        try:
            pattern = nltk.ParentedTree.fromstring("(NP (*))")
            tree = par_file_data[sent_index]
            subtree = pattern_matcher(pattern, tree)
            answer = " ".join(subtree.leaves())
            return str(answer)
        except:
            #pattern = nltk.ParentedTree.fromstring("(VP (*))")
            pattern = nltk.ParentedTree.fromstring("(S)")
            tree = par_file_data[sent_index]
            subtree = pattern_matcher(pattern, tree)
            answer = " ".join(subtree.leaves())
            return str(answer)


    elif wh_word == "Where":
        try:
            pattern = nltk.ParentedTree.fromstring("(PP (*))")
            tree = par_file_data[sent_index]
            subtree = pattern_matcher(pattern, tree)
            answer = " ".join(subtree.leaves())
            return str(answer)
        except:
            pattern = nltk.ParentedTree.fromstring("(VP (*))")
            pattern = nltk.ParentedTree.fromstring("(S)")
            tree = par_file_data[sent_index]
            subtree = pattern_matcher(pattern, tree)
            answer = " ".join(subtree.leaves())
            return str(answer)

    elif wh_word == "When":
        try:
            pattern = nltk.ParentedTree.fromstring("(PP (*))")
            tree = par_file_data[sent_index]
            subtree = pattern_matcher(pattern, tree)
            answer = " ".join(subtree.leaves())
            return str(answer)
        except:
            pattern = nltk.ParentedTree.fromstring("(NP (*))")
            tree = par_file_data[sent_index]
            subtree = pattern_matcher(pattern, tree)
            answer = " ".join(subtree.leaves())
            return str(answer)

    elif wh_word == "Why":
        try:
            pattern = nltk.ParentedTree.fromstring("(ADJP (*))")
            tree = par_file_data[sent_index]
            subtree = pattern_matcher(pattern, tree)
            answer = " ".join(subtree.leaves())
            return str(answer)
        except:
            pattern = nltk.ParentedTree.fromstring("(VP (*))")
            tree = par_file_data[sent_index]
            subtree = pattern_matcher(pattern, tree)
            answer = " ".join(subtree.leaves())
            return str(answer)

    elif wh_word == "How":

        try:
            pattern = nltk.ParentedTree.fromstring("(ADVP) (*)")
            tree = par_file_data[sent_index]
            subtree = pattern_matcher(pattern, tree)
            answer = " ".join(subtree.leaves())
            return str(answer)
        except:
            pattern = nltk.ParentedTree.fromstring("(S)")
            tree = par_file_data[sent_index]
            subtree = pattern_matcher(pattern, tree)
            answer = " ".join(subtree.leaves())
            return str(answer)

    #print("question"+question)
    #print("index "+str(sent_index))

    tree = par_file_data[sent_index]
    subtree = pattern_matcher(pattern, tree)
    answer = " ".join(subtree.leaves())
    return str(answer)


def load_wordnet_ids(filename):
    file = open(filename, 'r')
    if "noun" in filename: type = "noun"
    else: type = "verb"
    csvreader = csv.DictReader(file, delimiter=",", quotechar='"')
    word_ids = defaultdict()
    for line in csvreader:
        word_ids[line['synset_id']] = {'synset_offset': line['synset_offset'], 'story_'+type: line['story_'+type], 'stories': line['stories']}
    return word_ids

#Processes hard questions
def process_question(question):
    noun_ids = load_wordnet_ids("Wordnet_nouns.csv")
    verb_ids = load_wordnet_ids("Wordnet_verbs.csv")

    nouns_verbs = []
    for nID, items in noun_ids.items():
        nouns_verbs.append(nID)

    for vID, items in verb_ids.items():
        nouns_verbs.append(vID)


    story_nouns = []
    story_verbs = []

    for synset_id, items in noun_ids.items():
        story_nouns.append(items['story_noun'])

    for synset_id, items in verb_ids.items():
        story_verbs.append(items['story_verb'])



    query = get_sentences(question)
    temp = []
    for sent in query:
        temp.extend(sent)

    #Appending all the new words that are not already present in the story
    noun_list = ["NN", "NNP", "NNS"]
    nouns = [word for (word, tag) in temp if tag in noun_list and word not in story_nouns]
    verb_list = ['VB', 'VBZ', 'VBN']
    verbs = [word for (word, tag) in temp if tag in verb_list and word not in story_verbs]
    wrong_tagged_words = ["notice" ,"witness" ,"strike" ,"gobble" ,"consume" ,"descend" ,"serve", "ignite" , "unloosen" ,"perceive", "occuring", "combusted", "gnawed"]
    for word in wrong_tagged_words:
       if word in nouns:
            nouns.remove(word)
            verbs.append(word)


    print("Nouns {}".format(nouns))
    print("Verbs {}".format(verbs))

    replacement_nouns = []
    for noun_word in nouns:
        #print(noun_word)
        replacement_nouns = []
        replacement_nouns.append(noun_word)
        word_synsets = wn.synsets(noun_word)

        #Synsets
        for word_synset in word_synsets:
            for nounID, items in noun_ids.items():
                if word_synset.name() in nounID:
                    replacement_nouns.append(word_synset.name())
                    replacement_nouns.append(items['story_noun'])
                    break
                else:
                    continue

        #Hyponyms
        for word_synset in word_synsets:
            word_hypo = word_synset.hyponyms()
            for hypo in word_hypo:
                for nounID, items in noun_ids.items():
                    if hypo.name() in nounID:
                        replacement_nouns.append(hypo.name())
                        replacement_nouns.append(items['story_noun'])
                        break
                    else:
                        continue

        #Hypernyms
        for word_synset in word_synsets:
            word_hyper = word_synset.hypernyms()
            for hyper in word_hyper:
                for nounID, items in noun_ids.items():
                    if hyper.name() in nounID:
                        replacement_nouns.append(hyper.name())
                        replacement_nouns.append(items['story_noun'])
                        break
                    else:
                        continue

    replacement_verbs = []
    for verb_word in verbs:
        #print(verb_word)
        replacement_verbs = []
        replacement_verbs.append(verb_word)
        word_synsets = wn.synsets(verb_word)

        #Synset
        for word_synset in word_synsets:
            for verbID, items in verb_ids.items():
                if word_synset.name() in verbID:
                    replacement_verbs.append(word_synset.name())
                    replacement_verbs.append(items['story_verb'])
                    break
                else:
                    continue

        #Hyponyms
        for word_synset in word_synsets:
            word_hypo = word_synset.hyponyms()
            for hypo in word_hypo:
                for verbID, items in verb_ids.items():
                    if hypo.name() in verbID:
                        replacement_verbs.append(hypo.name())
                        replacement_verbs.append(items['story_verb'])
                        break
                    else:
                        continue

        #Hypernyms
        for word_synset in word_synsets:
            word_hyper = word_synset.hypernyms()
            for hyper in word_hyper:
                for verbID, items in verb_ids.items():
                    if hyper.name() in verbID:
                        replacement_verbs.append(hyper.name())
                        replacement_verbs.append(items['story_verb'])
                        break
                    else:
                        continue

    #print(replacement_nouns)
    #print(replacement_verbs)

    try:
        if replacement_nouns[0] and replacement_nouns[2]:
            question = question.replace(replacement_nouns[0], replacement_nouns[2])
    except:
        question = question

    try:
        if replacement_verbs[0] and replacement_verbs[2]:
            #print("True")
            #print(question)
            question = question.replace(replacement_verbs[0], replacement_verbs[2])
            #print(question)
    except:
        question = question

    return question



if __name__ == "__main__":
    #Add parser here
    output_file = open("train_my_answers.txt", "w", encoding="utf-8")
    stopwords = set(nltk.corpus.stopwords.words("english"))
    cname_size_dict = OrderedDict();
    noun_counter = 0
    verb_counter = 0
    #This part changes. The value of the key changes to number of fables and blog files that are in the process-stories.txt
    #cname_size_dict.update({"fables": 6})
    #cname_size_dict.update({"blogs": 6})
    cname_size_dict.update({"blogs": 6})
    cname_size_dict.update({"fables": 6})

    for cname, size in cname_size_dict.items():
        for i in range(0, size):
            fname = "{0}-{1:02d}".format(cname, i + 1)
            data_dict = get_data_dict(fname)
            print(fname)
            par_data_story = read_con_parses(fname, "story")
            par_data_sch = read_con_parses(fname, "sch")

            #Getting the questions
            questions = getQA("{}.questions".format(fname))

            for j in range(0, len(questions)):
                qname = "{0}-{1}".format(fname, j + 1)
                if qname in questions:
                    print("QuestionID: " + qname)
                    question = questions[qname]['Question']
                    print(question)
                    qtypes = questions[qname]['Type']
                    qdiff = questions[qname]['Difficulty']
                    print(qdiff)

                    #Read the content of fname.questions.par, fname.questions.dep for hint
                    #question_dep = data_dict["questions.dep"]
                    #question_par = data_dict["questions.par"]

                    for qt in qtypes.split("|"):
                        qt = qt.strip().lower()

                        #These are the text data where you can look for answers
                        raw_text = data_dict[qt]
                        #par_text = data_dict[qt + ".par"]
                        #dep_text = data_dict[qt + ".dep"]


                        #Finding baseline answer here, initially

                        if qdiff == "Easy":
                            print(qt)
                            candidate_answers = get_candidate_answers(question, raw_text)
                            answer = candidate_answers[0][2]
                            index = candidate_answers[0][1]
                            if qt == "story":
                                answer = get_con_answers(question, par_data_story, index)
                            elif qt == "sch":
                                answer = get_con_answers(question, par_data_sch, index)

                            #print(qt)


                        elif qdiff == "Medium":

                            candidate_answers = get_candidate_answers(question, raw_text)
                            answer = candidate_answers[0][2]
                            index = candidate_answers[0][1]
                            #print(index)
                            print(qt)
                            if qt == "story":
                                answer = get_con_answers(question, par_data_story, index)
                            elif qt == "sch":
                                answer = get_con_answers(question, par_data_sch, index)

                        elif qdiff == "Hard":
                            question = process_question(question)
                            candidate_answers = get_candidate_answers(question, raw_text)
                            answer = candidate_answers[0][2]
                            index = candidate_answers[0][1]
                            print(question)
                            # print(index)
                            print(qt)
                            if qt == "story":
                                answer = get_con_answers(question, par_data_story, index)
                            elif qt == "sch":
                                answer = get_con_answers(question, par_data_sch, index)

                    print("Answer: " + str(answer))
                    print("")

                    output_file.write("QuestionID: {}\n".format(qname))
                    output_file.write("Answer: {}\n\n".format(answer))

                    #main ends here
    #print(par_text)
    #print(par_data_sch)

    #print(par_data_story)
    output_file.close()
