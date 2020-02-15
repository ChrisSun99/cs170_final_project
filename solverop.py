import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils
from student_utils import *
from sys import maxsize



"""
======================================================================
  Complete the following function.
======================================================================
"""

max_iter = 200
n_clusters = 10

def calculate_count(list_of_locations):
    if len(list_of_locations) < 51:
        return 8
    elif len(list_of_locations) < 101:
        return 6
    else:
        return 4

def convert_to_index(homes, locations):

    new_homes = np.zeros(homes.shape, dtype=int)
    for i in range(len(locations)):
        indices = np.argwhere(homes == locations[i])
        np.put(new_homes, indices, i)
    return new_homes

def all_pair_distance(data):
    dist = np.copy(data)
    for k in range(data.shape[0]):
        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                if i == j:
                    dist[i][j] = 0
                else:
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist

def initializ_centroids(dist, homes):
    np.random.RandomState(123)
    random_idx = np.random.permutation(homes)
    centroids = dist[random_idx[:10]]

    return centroids

def find_closest_cluster(centroids):
    clusters = np.argmin(centroids, axis=0)
    return clusters

def compute_distance(dist, centroids):
    centers = np.argmin(centroids, axis=1).flatten()
    new_distance = dist[centers]
    return new_distance, centers

def compute_centroids(dist, clusters, n_clusters, homes):
    centroids = np.zeros((n_clusters, dist.shape[0]))
    for k in range(n_clusters):
        tmp = np.argwhere(clusters == k).flatten()
        tmp = tmp[np.in1d(tmp, homes)]
        centroids[k, :] = np.mean(dist[tmp], axis=0)
    return centroids

def translate(clusters, random_idx):
    new_clusters = np.zeros(clusters.shape, dtype=int)
    for i in range(len(random_idx[:10])):
        indices = np.argwhere(clusters == i)
        np.put(new_clusters, indices, random_idx[i])
    return new_clusters

def fit(dist, homes):
    centroids = initializ_centroids(dist, homes)
    for i in range(max_iter):
        old_centroids = centroids
        distance, centers = compute_distance(dist, old_centroids)
        clusters = find_closest_cluster(distance)
        centroids = compute_centroids(dist, clusters, n_clusters, homes)
        if np.array_equal(old_centroids, centroids):
            break
    new_clusters = translate(clusters, centers)
    return np.argmin(centroids, axis=1), new_clusters

def sanity_check(dist_matrix, clusters, centers, starting_location):
    sd = np.std(dist_matrix)
    mean = np.mean(dist_matrix)
    check = np.zeros(dist_matrix.shape[0])
    for i in range(dist_matrix.shape[0]):
        check[i] = dist_matrix[i][clusters[i]]
    for i in range(len(check)):
<<<<<<< HEAD
        if (check[i] > mean + 2.5 * sd and len(centers) < 11):
=======
        if (check[i] > mean + 2.5 * sd and len(centers) < 13):
>>>>>>> e6ca4e0a157553c94a0d81e4a31a8e46457bccfe
            clusters[i] = i
            centers = np.concatenate((centers, [i]), axis=0)
    if starting_location not in centers:
        centers = np.concatenate((centers, [starting_location]), axis=0)
        clusters[starting_location] = starting_location
    return centers, clusters

# implementation of traveling Salesman Problem
def travellingSalesmanProblem(graph, s, path):
    # store all vertex apart from source vertex
    vertex = []
    tmp = path
    for i in range(graph.shape[0]):
        if i != s:
            vertex.append(i)
    # store minimum weight Hamiltonian Cycle
    min_path = maxsize
    while True:
        # store current Path weight(cost)
        current_pathweight = 0
        # compute current path weight
        k = s
        for i in range(len(vertex)):
            current_pathweight += graph[k][vertex[i]]
            path.append(k)
            k = vertex[i]
        path.append(k)
        current_pathweight += graph[k][s]
        # update minimum
        if min_path > current_pathweight:
            min_path = current_pathweight
            tmp = path
        path = []
        if not next_permutation(vertex):
            break
    return min_path, tmp

# next_permutation implementation
def next_permutation(L):
    n = len(L)
    i = n - 2
    while i >= 0 and L[i] >= L[i + 1]:
        i -= 1
    if i == -1:
        return False
    j = i + 1
    while j < n and L[j] > L[i]:
        j += 1
    j -= 1
    L[i], L[j] = L[j], L[i]
    left = i + 1
    right = n - 1
    while left < right:
        L[left], L[right] = L[right], L[left]
        left += 1
        right -= 1
    return True


def dijsktra(dist, initial, end, homes):
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()

    while current_node != end:
        visited.add(current_node)
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in range(len(dist)):
            weight = weight_to_current_node + dist[current_node][next_node]
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)

        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not np.array_equal(homes, np.array([])):
            for node in next_destinations:
                if node in homes:
                    a, b = next_destinations[node]
                    next_destinations[node] = (a, 1/6*b)

        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

    # Work back through destinations in shortest path
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    return path

##----------------------------------------------------------------------------##
#small triangle optimization
def tri_approx(adjacency_matrix, dist_matrix, homes, path):
    for i in range(len(homes)):
        if (not np.isin(homes[i], path)):
            dist_arr = dist_matrix[homes[i]]
            mindir = np.inf
            for k in range(len(dist_arr)):
                if (k != homes[i] and np.isin(k, path) and dist_arr[k] < mindir):
                    mindir = dist_arr[k]
            index = len(path)
            min = np.inf
            for j in range(len(path)-1):
                tmp_dist = (2/3)*(dist_arr[path[j]]+dist_arr[path[j+1]])
                if (tmp_dist < ((2/3)*adjacency_matrix[path[j]][path[j+1]] + mindir) and tmp_dist < min):
                    index = j
                    min = tmp_dist
            if (index != len(path)):
                first = path[:index]
                second = path[index+2:]
                amendment1 = dijsktra(adjacency_matrix, path[index], homes[i], np.array([]))
                amendment2 = dijsktra(adjacency_matrix, homes[i], path[index+1], np.array([]))
                amendment = np.concatenate((amendment1, amendment2[1:]))
                first = np.concatenate((first, amendment))
                path = np.concatenate((first, second))
                path = np.array(path, dtype=int)
    return path

#small square optimization
def sqr_approx(adjacency_matrix, dist_matrix, homes, path):
    for i in range(len(homes)):
        for j in range(i+1, len(homes)):
            if ((not np.isin(homes[i], path)) and (not np.isin(homes[j], path))):
                dist_arr = dist_matrix[homes[i]]
                mindir = np.inf
                for m in range(len(dist_arr)):
                    if (m != homes[i] and np.isin(m, path) and dist_arr[m] < mindir):
                        mindir = dist_arr[m]
                tmp_dist_arr = dist_matrix[homes[j]]
                mindir_second = np.inf
                for m in range(len(tmp_dist_arr)):
                    if (m != homes[j] and np.isin(m, path) and tmp_dist_arr[m] < mindir_second):
                        mindir_second = tmp_dist_arr[m]
                min = np.inf
                index = len(path)
                for k in range(len(path)-1):
                    tmp_dist = dist_arr[path[k]] + dist_matrix[homes[i]][homes[j]] + tmp_dist_arr[path[k+1]]
                    tmp_dist = tmp_dist * 2 /3
                    if (tmp_dist < ((2/3)*adjacency_matrix[path[k]][path[k+1]] + mindir + mindir_second) and tmp_dist < min):
                        index = k
                        min = tmp_dist
                if (index != len(path)):
                    first = path[:index]
                    second = path[index+2:]
                    amendment1 = dijsktra(adjacency_matrix, path[index], homes[i], np.array([]))
                    amendment2 = dijsktra(adjacency_matrix, homes[i], homes[j], np.array([]))
                    amendment3 = dijsktra(adjacency_matrix, homes[j], path[index+1], np.array([]))
                    amendment = np.concatenate((amendment1, amendment2[1:]))
                    amendment = np.concatenate((amendment, amendment3[1:]))
                    first = np.concatenate((first, amendment))
                    path = np.concatenate((first, second))
                    path = np.array(path, dtype=int)
    return path

#round-trip optimization
def roundtrip_reduction(real_path, old_drop_off_mapping):
    path = np.copy(real_path)
    drop_off_mapping = old_drop_off_mapping.copy()
    i = len(path)-2
    while(i > 0):
        if (path[i-1] == path[i+1] and path[i] in drop_off_mapping):
            drops = drop_off_mapping[path[i]]
            if (len(drops) == 1):
                if path[i-1] in drop_off_mapping:
                    drop_off_mapping[path[i-1]] = drop_off_mapping[path[i-1]] + drops
                else:
                    drop_off_mapping[path[i-1]] = drops
                del drop_off_mapping[path[i]]
                path = np.concatenate((path[:i-1], path[i+1:]))

            i = i-2
        else:
            i = i-1
    return path, drop_off_mapping


##----------------------------------------------------------------------------##

def real_solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix):
    """
    Write your algorithm here.
    Input:
        list_of_locations: A list of locations such that node i of the graph corresponds to name at index i of the list
        list_of_homes: A list of homes
        starting_car_location: The name of the starting location for the car
        adjacency_matrix: The adjacency matrix from the input file
    Output:
        A list of locations representing the car path
        A dictionary mapping drop-off location to a list of homes of TAs that got off at that particular location
        NOTE: both outputs should be in terms of indices not the names of the locations themselves
    """

    # find the clusters and centers
    homes = convert_to_index(np.array(list_of_homes), np.array(list_of_locations))
    adjacency_matrix = np.array(adjacency_matrix)
    adjacency_matrix = np.where(adjacency_matrix == 'x', np.inf, adjacency_matrix)
    adjacency_matrix = np.array(adjacency_matrix, dtype=float)
    dist_matrix = all_pair_distance(np.array(adjacency_matrix))


    centers, clusters = fit(dist_matrix, homes)
    s = list_of_locations.index(starting_car_location)
    centers, clusters = sanity_check(dist_matrix, clusters, centers, s)

    # calculate the tsp path
    graph = dist_matrix[centers, :][:, centers]
    tsp_path = []
    min_path, tsp_path = travellingSalesmanProblem(graph, 0, tsp_path)
    for i in range(len(tsp_path)):
        tsp_path[i] = centers[tsp_path[i]]
    tsp_path = np.roll(tsp_path, -tsp_path.index(s))
    tsp_path = np.concatenate((tsp_path, [tsp_path[0]]))
    print(tsp_path)

    # find the real path using dijsktra
    real_path = []
    for i in range(len(tsp_path) - 1):
        real_path = real_path + dijsktra(adjacency_matrix, tsp_path[i], tsp_path[i + 1], homes)[:-1]
    real_path = real_path + [tsp_path[0]]

    #-----------------------------
    opiter = 5
    while (opiter != 0):
        tmp_path_copy = np.copy(real_path)
        real_path = tri_approx(adjacency_matrix, dist_matrix, homes, real_path)
        real_path = sqr_approx(adjacency_matrix, dist_matrix, homes, real_path)
        if (np.array_equal(tmp_path_copy, real_path)):
            break
        opiter -= 1

    #-----------------------------


    # construct the drop_off_mapping
    helper = dist_matrix[real_path]
    mins = np.argmin(helper, axis=0)
    result = np.zeros(mins.shape)
    for i in range(len(real_path)):
        indices = np.argwhere(mins == i)
        np.put(result, indices, real_path[i])
    for i in range(len(result)):
        if i not in homes:
            result[i] = -1
    drop_off_mapping = {}
    for i in range(len(real_path)):
        drop_off_mapping[real_path[i]] = np.argwhere(result == real_path[i]).flatten().tolist()
        if not drop_off_mapping[real_path[i]]:
            del drop_off_mapping[real_path[i]]
    #roundtrip_reduction
    real_path1, drop_off_mapping1 = roundtrip_reduction(real_path, drop_off_mapping)
    #
    return real_path, drop_off_mapping, real_path1, drop_off_mapping1


def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    min_cost = np.inf
    G, message = adjacency_matrix_to_graph(adjacency_matrix)
    real_path = []
    drop_off_mapping = {}
    reduction_flag = 0
    count = calculate_count(list_of_locations)
    for i in range(count):
        tmp_path, tmp_drop_off_mapping, tmp_path1, tmp_drop_off_mapping1 = real_solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix)
        cost, solution_message = cost_of_solution(G, tmp_path, tmp_drop_off_mapping)
        cost1, solution_message1 = cost_of_solution(G, tmp_path1, tmp_drop_off_mapping1)
        if (cost < min_cost):
            min_cost = cost
            real_path = tmp_path
            drop_off_mapping = tmp_drop_off_mapping
            reduction_flag = 1
        if (cost1 < min_cost):
            min_cost = cost1
            real_path = tmp_path1
            drop_off_mapping = tmp_drop_off_mapping1
            reduction_flag = 2
    print(min_cost)
    return real_path, drop_off_mapping


"""
======================================================================
   No need to change any code below this line
======================================================================
"""

"""
Convert solution with path and dropoff_mapping in terms of indices
and write solution output in terms of names to path_to_file + file_number + '.out'
"""
def convertToFile(path, dropoff_mapping, path_to_file, list_locs):
    string = ''
    for node in path:
        string += list_locs[node] + ' '
    string = string.strip()
    string += '\n'

    dropoffNumber = len(dropoff_mapping.keys())
    string += str(dropoffNumber) + '\n'
    for dropoff in dropoff_mapping.keys():
        strDrop = list_locs[dropoff] + ' '
        for node in dropoff_mapping[dropoff]:
            strDrop += list_locs[node] + ' '
        strDrop = strDrop.strip()
        strDrop += '\n'
        string += strDrop
    utils.write_to_file(path_to_file, string)

def solve_from_file(input_file, output_directory, params=[]):
    print('Processing', input_file)

    input_data = utils.read_file(input_file)
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    car_path, drop_offs = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    basename, filename = os.path.split(input_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file = utils.input_to_output(input_file, output_directory)

    convertToFile(car_path, drop_offs, output_file, list_locations)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')
    counter = 0
    for input_file in input_files:
        counter += 1
        print(counter, "out of 90")
        solve_from_file(input_file, output_directory, params=params)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--all', action='store_true', help='If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file')
    parser.add_argument('input', type=str, help='The path to the input file or directory')
    parser.add_argument('output_directory', type=str, nargs='?', default='.', help='The path to the directory where the output should be written')
    parser.add_argument('params', nargs=argparse.REMAINDER, help='Extra arguments passed in')
    args = parser.parse_args()
    output_directory = args.output_directory
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)
