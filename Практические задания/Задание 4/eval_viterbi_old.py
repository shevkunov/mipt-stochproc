from tqdm import tqdm_notebook, tnrange
from tqdm import tqdm
def my_log(f):
    if f != 0:
        return np.log(f)
    else:
        return 0. - np.inf
        
        
def eval_viterbi(y, log=False):
    T = len(y)

    g1 = np.zeros((len(tags), len(tags)))  # g1 из презентации для вершины x = (t1, t2) 
    for t1 in range(len(tags)):
        for t2 in range(len(tags)):
            g1[t1, t2] = (my_log(start_probs[t2]) +
                         my_log(condition_prob_matrix[y[0], t2]))

    g = np.zeros((T, len(tags), len(tags), len(tags)))
    # g из презентации, где g[T, t1, t2, t3] соответствует переходу из 
    # состояния x1 = (t1, t2) в состояние x2 = (t2, t3) во время T (y[T])
    # это нужно, т.к. переходы из (a, b) в (c,d), где b != c невозможны,
    # а хранение лишней памяти - растратно и медленно (промахи кэша и выделение памяти)
    
    """
    for t1 in range(len(tags)):
        for t2 in range(len(tags)):
            for t3 in range(len(tags)):
                g[0, t1, t2, t3] = g1[t2, t3]
    """ 
    # Код снизу делает то же самое, что и код сверху
    for t1 in range(len(tags)):
        g[0, t1, :, :] = g1
    
    
    for t in range(T):
        if (log):
            print(t, end=",")
        for x2 in range(r):
            t2, t3 = decode_tag_pair(x2)
            v2 = my_log(condition_prob_matrix[y[t], t3])
            for t1 in range(len(tags)):
                x1 = encode_tag_pair(t1, t2)
                v1 = my_log(prob_matrix[t1, t2, t3])
                g[t, t1, t2, t3] = v1 + v2
                
                
    ################
    if (log):
        print("G_dp")
        
    G_dp_vals = np.zeros((T, r))
    G_dp_ways = np.zeros((T, r), dtype=int)
    for k in range(r):  # инициализация ДП
        t2, t3 = decode_tag_pair(k)
        G_dp_vals[0, k] = g1[t2, t3]
        G_dp_ways[0, k] = -1 # храним пару - значение и номер ячейки, откуда пришли

    for t in range(1, T):
        
        if (log):
            print(t, end=",")
            
        for k in range(r):
            t2, t3 = decode_tag_pair(k)
            maximal = (G_dp_vals[t - 1, encode_tag_pair(0, t2)] + g[t, 0, t2, t3], encode_tag_pair(0, t2))
            for t1 in range(len(tags)):
                l = encode_tag_pair(t1, t2)
                var = (G_dp_vals[t - 1, l] + g[t, t1, t2, t3], l)
                if (maximal < var):
                    maximal = var
            G_dp_vals[t, k] = maximal[0]
            G_dp_ways[t, k] = maximal[1]
            
    #################
    
    k_star = G_dp_vals[T - 1, :].argmax()
    
    if (log):
        print((G_dp_vals[T-1, :] != -np.inf).sum())
        
    ans = [-1] * T
    for t in range(T)[::-1]:
        ans[t] = decode_tag_pair(k_star)[1]
        k_star = G_dp_ways[t, k_star]

    return ans

def tag(sent):
    y = []
    for word in sent:
        if word.lower() in words:
            y.append(words[word.lower()])
        else:
            print("fuck")
            y.append(words["this"])  # TODO заменить на что-то более разумное
    print("y = ", y)
    eval_viterbi(y)

def score_me(test_sent, log=False):
    if log:
        print("Score_me = ", test_sent)
    sent = [word for word, tag in test_sent]
    sent_tag = [tag for word, tag in test_sent]

    y = []
    for word in sent:
        if word.lower() in words:
            y.append(words[word.lower()])
        else:
            if log:
                print("fuck:", word)
            y.append(words["this"])  # TODO заменить на что-то более разумное
    if log:
        print("y = ", y)

    X = []
    for tag in sent_tag:
        X.append(tags[tag]) 
    if log:
        print("X = ", X)
                
    X_predict = eval_viterbi(y)
    if log:
        print("X_predict = ", X_predict)
    acc_score = accuracy_score(X_predict, X)
    if log:
        print("Accuracy = ", acc_score)
    return ((np.array(X_predict) == np.array(X)).sum(), len(X))
                
