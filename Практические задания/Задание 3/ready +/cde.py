import numpy as np

def create_page_rank_markov_chain(links, damping_factor=0.15, N=None):
    ''' По веб-графу со списком ребер links строит матрицу 
    переходных вероятностей соответствующей марковской цепи.
    
        links --- список (list) пар вершин (tuple), 
                может быть передан в виде numpy.array, shape=(|E|, 2);
        damping_factor --- вероятность перехода не по ссылке (float);
        
        Возвращает prob_matrix --- numpy.matrix, shape=(|V|, |V|).
    '''

    links = np.array(links)
    if N is None:
        N = links.max() + 1  # Число веб-страниц  <-- НЕТ !!!
        # Не работает, когда последняя вершина изолирована!!!
        # для этого был добавлен параметр
    
    ### begin code
    
    edge_matrix = np.zeros((N, N), dtype=int)  # создадим матрицу смежности
    for edge in links:
        edge_matrix[edge[0], edge[1]] = 1  # мы запрещаем кратные рёбра, согласно определению выше
    
    prob_matrix = np.zeros((N, N), dtype=float)  # матрица переходных вероятностей
    for vertex_id in range(N):
        outgoing_links = edge_matrix[vertex_id, :].sum()  # Ni в старых терминах
        if (outgoing_links == 0):
            prob_matrix[vertex_id, :] = np.ones(N, dtype=float) / N
        else:
            for dest_id in range(N):
                prob_matrix[vertex_id, dest_id] = (
                    damping_factor / N +
                    (1. - damping_factor) / outgoing_links * edge_matrix[vertex_id, dest_id]
                )
            
            
    ### end code
    
    return prob_matrix

def page_rank(links, start_distribution, damping_factor=0.15, 
              tolerance=10 ** (-7), return_trace=False, N=None):
    ''' Вычисляет веса PageRank для веб-графа со списком ребер links 
    степенным методом, начиная с начального распределения start_distribution, 
    доводя до сходимости с точностью tolerance.
    
        links --- список (list) пар вершин (tuple), 
                может быть передан в виде numpy.array, shape=(|E|, 2);
        start_distribution --- вектор размерности |V| в формате numpy.array;
        damping_factor --- вероятность перехода не по ссылке (float);
        tolerance --- точность вычисления предельного распределения;
        return_trace --- если указана, то возвращает список распределений во 
                            все моменты времени до сходимости
    
        Возвращает:
        1). если return_trace == False, то возвращает distribution --- 
        приближение предельного распределения цепи,
        которое соответствует весам PageRank.
        Имеет тип numpy.array размерности |V|.
        2). если return_trace == True, то возвращает также trace ---
        список распределений во все моменты времени до сходимости. 
        Имеет тип numpy.array размерности 
        (количество итераций) на |V|.
    '''
    prob_matrix = create_page_rank_markov_chain(links, 
                                                damping_factor=damping_factor, N=N)
    distribution = np.matrix(start_distribution)
    
    ### begin code
    trace = [distribution]  # включаем нулевую итерацию
    while True:
        new_distribution = distribution @ prob_matrix
        trace.append(new_distribution)

        if (np.linalg.norm(new_distribution - distribution, ord=2) <= tolerance):
            break;
        
        distribution = new_distribution
    ### end code
    
    if return_trace:
        return np.array(distribution).ravel(), np.array(trace)
    else:
        return np.array(distribution).ravel()
