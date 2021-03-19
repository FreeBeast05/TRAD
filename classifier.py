import random
def read_data():
    input_file = open('file_df_1', 'r')
    global data
    global labels
    data = {}
    labels = {}
    key_obj = 0
    for row in input_file:
        str_list = row.strip('\n').split(',')
        labels[key_obj] = int(str_list[-1])
        data[key_obj] = [int(i) for i in str_list[0:-1]]
        key_obj += 1
    input_file.close()
    return data, labels


def closed_concept(attributes, context, num_attributes):
    objects = []
    attributes_closed = attributes[:]

    for obj in context.keys():
        flag = True
        for attr in attributes:
            if context[obj][attr] == 0:
                flag = False
                break
        if flag:
            objects.append(obj)
    for attr in set(range(num_attributes))-set(attributes):
        flag = True
        for obj in objects:
            if context[obj][attr] == 0:
                flag = False
                break
        if flag:
            attributes_closed.append(attr)

    return objects[:], attributes_closed[:]

def ganter_algorithm(context, num_attributes):

    lattice = {}
    attributes = []
    formal_notion= closed_concept(attributes, context, num_attributes)
    attributes = formal_notion[1][:]
    lattice[0] = formal_notion
    index_current_concept = 0
    while len(attributes) < num_attributes:
        for attr in range(num_attributes-1, -1, -1):
            if attr in set(attributes):
                attributes.remove(attr)
            else:
                attributes.append(attr)
                formal_notion = closed_concept(attributes, context, num_attributes)
                flag_is_new_concept = True
                for new_attr in set(formal_notion[1])-set(attributes):
                    if new_attr < attr:
                        attributes.remove(attr)
                        flag_is_new_concept = False
                        break

                if flag_is_new_concept:
                    index_current_concept += 1
                    lattice[index_current_concept] = formal_notion
                    attributes = formal_notion[1][:]
                    break

    return lattice
def scheme1_h(train_positive, train_negative, lattice_positive, lattice_negative, percent_bad):
    num_attributes = len(train_positive[list(train_positive.keys())[0]])

    h_p = {}
    i_h_p = 0
    for key, concept in lattice_positive.items():
        num_contradictions = 0
        for key2, row in train_negative.items():
            row_attr = [i for i in range(num_attributes) if row[i] == 1]
            if set(concept[1]).issubset(set(row_attr)):
                num_contradictions += 1

        if num_contradictions < percent_bad*len(concept[0]):
            h_p[i_h_p] = concept
            i_h_p += 1

    h_n = {}
    i_h_n = 0
    for key, concept in lattice_negative.items():
        num_contradictions = 0
        for key2, row in train_positive.items():
            row_attr = [i for i in range(num_attributes) if row[i] == 1]
            if set(concept[1]).issubset(set(row_attr)):
                num_contradictions += 1

        if num_contradictions < percent_bad*len(concept[0]):
            h_n[i_h_n] = concept
            i_h_n += 1

    return h_p, h_n

def scheme1_a(test_data, h_p, h_n, percent_aggr):
    num_attributes = len(test_data[list(test_data.keys())[0]])

    future_test_posit = []
    future_test_negat = []

    for key, row in test_data.items():
        row_attr = set([i for i in range(num_attributes) if row[i] == 1])

        h_p_indices = set()
        for h_ind, concept in h_p.items():
            if set(concept[1]).issubset(row_attr):
                h_p_indices_new = set()
                for h_ind2 in h_p_indices:
                    if ~set(concept[1]).issuperset(set(h_p[h_ind2][1])):
                        h_p_indices_new.add(h_ind2)
                h_p_indices_new.add(h_ind)
                h_p_indices = h_p_indices_new.copy()

        h_n_indices = set()
        for h_ind, concept in h_n.items():
            if set(concept[1]).issubset(row_attr):
                h_n_indices_new = set()
                for h_ind2 in h_n_indices:
                    if ~set(concept[1]).issuperset(set(h_n[h_ind2][1])):
                        h_n_indices_new.add(h_ind2)
                h_n_indices_new.add(h_ind)
                h_n_indices = h_n_indices_new.copy()

        obj_positive = set()
        obj_negative = set()
        for h_ind in h_p_indices:
            obj_positive |= set(h_p[h_ind][0])

        for h_ind in h_n_indices:
            obj_negative |= set(h_n[h_ind][0])

        num_obj_positive = len(obj_positive)
        num_obj_negative = len(obj_negative)

        if num_obj_positive < percent_aggr*num_obj_negative:
            future_test_negat.append(key)
        elif num_obj_negative < percent_aggr*num_obj_positive:
            future_test_posit.append(key)

    return future_test_posit, future_test_negat

def scheme2_a(test_data, h_p, h_n, percent_aggr):
    num_attributes = len(test_data[list(test_data.keys())[0]])

    future_test_posit = []
    future_test_negat = []

    for key, row in test_data.items():
        row_attr = set([i for i in range(num_attributes) if row[i] == 1])

        h_p_indices = set()
        for h_ind, concept in h_p.items():
            if set(concept[1]).issubset(row_attr):
                h_p_indices.add(h_ind)

        h_n_indices = set()
        for h_ind, concept in h_n.items():
            if set(concept[1]).issubset(row_attr):
                h_n_indices.add(h_ind)

        num_obj_positive = len(h_p_indices)
        num_obj_negative = len(h_n_indices)

        if num_obj_positive < percent_aggr*num_obj_negative:
            future_test_negat.append(key)
        elif num_obj_negative < percent_aggr*num_obj_positive:
            future_test_posit.append(key)

    return future_test_posit, future_test_negat

def classify_test_data(train_data, test_data, labels, koif):
    num_attributes = len(test_data[list(test_data.keys())[0]])
    label_positive = 1
    label_negative = 0
    sample = {label_positive: [], label_negative: []} # 1(позитив) или 0 (негатив) в целевом признаке
    for i, l in labels.items():
        if l == label_positive:
            sample[label_positive].append(i)
        elif l == label_negative:
            sample[label_negative].append(i)
        else:
            print('Label error', l, i)

    train_positive = {}
    train_negative = {}
    for obj in sample[label_positive]:
        train_positive[obj] = train_data[obj]
    for obj in sample[label_negative]:
        train_negative[obj] = train_data[obj]
    lattice_positive = ganter_algorithm(train_positive, num_attributes)
    lattice_negative = ganter_algorithm(train_negative, num_attributes)

    h_p, h_n = scheme1_h(train_positive, train_negative, lattice_positive, lattice_negative, koif )
    return scheme1_a(test_data, h_p, h_n, koif)

data, labers= read_data()

train_portion = 0.8
train_ind = set(random.sample(range(len(data)), round(train_portion*len(data))))
test_ind = set(range(len(data))) - train_ind


train_data = {}
train_labels = {}
for ind in train_ind:
    train_data[ind] = data[ind]
    train_labels[ind] = labels[ind]

test_data = {}
for ind in test_ind:
    test_data[ind] = data[ind]

for i in [0.1, 0.2, 0.3,0.4, 0.5, 0.6, 0.7]:
    pos, neg = classify_test_data(train_data, test_data, train_labels, i)
    label_positive = 1
    label_negative = 0
    print(i)
    accuracy = 0
    for key in pos:
        if labels[key] == label_positive:
            accuracy += 1

    for key in neg:
        if labels[key] == label_negative:
            accuracy += 1
    print("количество верно классифиц", accuracy)
    print(accuracy, len(test_data))