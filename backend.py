from threading import local

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import random

app = Flask(__name__)
CORS(app, origins='*', supports_credentials=True)

class Resource:
    def __init__(self, name, count):
        self.name = name
        self.total = count
        self.available = count


class Process:
    def __init__(self, max_requirements, allocation):
        self.max_requirements = max_requirements
        self.allocation = allocation
        self.need = [m - a for m, a in zip(max_requirements, allocation)]
        self.finished = False


class System:
    def __init__(self):
        self.resources = []
        self.processes = []

    def add_resource(self, resource):
        self.resources.append(resource)

    def add_process(self, process):
        self.processes.append(process)


def banker_algorithm(system):
    safe_sequence = []
    total_allocations = [0] * len(system.resources)
    while len(safe_sequence) < len(system.processes):
        for i, process in enumerate(system.processes):
            if process.finished:
                continue
            if all(n <= system.resources[j].available for j, n in enumerate(process.need)):
                safe_sequence.append(i)
                process.finished = True
                for j, a in enumerate(process.allocation):
                    system.resources[j].available += a
                    total_allocations[j] += a
        if not any(process.finished == False for process in system.processes):
            break

    if all(process.finished for process in system.processes):
        deadlock_frequency = 0
        resource_utilization = [total_allocations[j] / system.resources[j].total for j in range(len(system.resources))]
        total_scheduling_operations = len(safe_sequence)
        return safe_sequence, deadlock_frequency, resource_utilization, total_scheduling_operations
    else:
        return None




@app.route('/banker', methods=['POST'])
def calculate_banker():
    data = request.json

    system0 = System()
    # Initialize resources
    for resource_data in data['resources']:
        system0.add_resource(Resource(resource_data['name'], resource_data['count']))

    # Initialize processes
    for process_data in data['processes']:
        system0.add_process(Process(process_data['max_requirements'], process_data['allocation']))

    result = banker_algorithm(system0)
    if result:
        safe_sequence, deadlock_frequency, resource_utilization, total_scheduling_operations = result
        return jsonify({
            "safe_sequence": safe_sequence,
            "deadlock_frequency": deadlock_frequency,
            "resource_utilization": resource_utilization,
            "total_scheduling_operations": total_scheduling_operations
        })
    else:
        return jsonify({"message": "No safe sequence found. System may deadlock."})





# 使用线程局部变量来代替全局变量
thread_local = local()


def Alloc_Num1():  # 分配结点编号的，依赖于环境变量cnt
    thread_local.cnt1 += 1
    return thread_local.cnt1


def Alloc_Num2():  # 分配结点编号的，依赖于环境变量cnt
    thread_local.cnt2 += 1
    return thread_local.cnt2


class Node:
    def __init__(self):
        self.num = Alloc_Num1()
        self.out_edges = []
        self.in_edges = []

    def add_out_edge(self, edge):
        self.out_edges.append(edge)

    def add_in_edge(self, edge):
        self.in_edges.append(edge)


class ResourceNode(Node):
    def __init__(self, resource_count):
        super().__init__()
        self.resource_count = resource_count


class ProcessNode(Node):
    def __init__(self, allocated, needed):
        super().__init__()
        self.allocated_resources = allocated
        self.needed_resources = needed
        self.resource_num = len(allocated)


class Edge:
    def __init__(self, from_node, to_node):
        self.from_node = from_node
        self.to_node = to_node
        self.num = Alloc_Num2()


def create_graph(r, p):
    resources=r[:]
    processes=p[:]
    n = len(resources)
    # resources = []
    for i in range(n):
        resource = resources[i]
        resource_count = resource['count']
        t = ResourceNode(resource_count)
        resources.append(t)
        thread_local.Nodes.append(t)

    m = len(processes)
    # processes = []

    for i in range(m):
        process = processes[i]
        allocated = process['allocation']
        needed = process['max_requirements']
        for k in range(n):
            for r in range(allocated[k]):
                # print(k,n+i)
                edge = Edge(k, n + i)
                thread_local.Graph.append(edge)
            for r in range(needed[k]):
                # print(n+i,k)
                edge = Edge(n + i, k)
                thread_local.Graph.append(edge)

        t = ProcessNode(allocated, needed)
        processes.append(t)
        thread_local.Nodes.append(t)

def showGraph(Graph):
    print("*************************")
    print("Graph edges:")
    for edge in Graph:
        print(f"Edge {edge.num}: From Node {edge.from_node} to Node {edge.to_node}")
    print("*************************")


def Step1():
    new_edge = []
    for edge in thread_local.Graph:
        u = thread_local.Nodes[edge.from_node]
        v = thread_local.Nodes[edge.to_node]
        if type(u) == ResourceNode:
            thread_local.Nodes[edge.from_node].resource_count -= 1;
            if thread_local.Nodes[edge.from_node].resource_count < 0:
                return False
            continue
        new_edge.append(edge)
    thread_local.Graph = new_edge
    return True


def findProcessOut(num):  # 判断进程可以满足资源数的
    # print("************")
    current_resource = []
    for i in range(len(thread_local.Nodes)):
        current_resource.append(0)
    for node in thread_local.Nodes:
        if type(node) == ResourceNode:
            current_resource[node.num] = node.resource_count
    for edge in thread_local.Graph:
        if edge.from_node == num:
            current_resource[edge.to_node] -= 1
    for x in current_resource:
        if x < 0:
            return False
    return True


def ReleaseProcess(num):
    new_Graph = []
    for edge in thread_local.Graph:
        u = edge.from_node
        v = edge.to_node
        if u != num and v != num:
            new_Graph.append(edge)
            continue
    for i in range(len(thread_local.Nodes[num].allocated_resources)):
        thread_local.Nodes[i].resource_count += thread_local.Nodes[num].allocated_resources[i]
    # for i in range(len(Nodes[num].allocated_resources)):
    #     print(Nodes[i].resource_count)
    thread_local.Graph = new_Graph
    # showGraph(new_Graph)


def Graph_Allocate_Algorithm():
    Graphs = []
    Graphs.append(thread_local.Graph)  # 原始图
    lock = 0
    # 划掉已经被分配走的资源
    if not Step1():
        return None
    Graphs.append(thread_local.Graph)
    showGraph(thread_local.Graph)
    while len(thread_local.Graph) >= 1:
        lock += 1
        if (lock > 1000):  # 暴力判断找不到可消去的点判断死锁
            return None
        # 找到满足进程的边并回收资源
        for edge in thread_local.Graph:
            u = thread_local.Nodes[edge.from_node]
            v = thread_local.Nodes[edge.to_node]
            if type(u) == ProcessNode and findProcessOut(edge.from_node):
                ReleaseProcess(edge.from_node)
                break
        Graphs.append(thread_local.Graph)
        showGraph(thread_local.Graph)
    return Graphs


@app.route('/allocate_resources', methods=['POST'])
def allocate_resources():
    # 从前端接收数据
    data = request.get_json()
    # print(type(data))
    resources = [resource for resource in data['resources']]
    processes = [process for process in data['processes']]
    # print(resources)
    # 初始化线程局部变量
    thread_local.cnt1 = -1
    thread_local.cnt2 = -1
    thread_local.Nodes = []
    thread_local.Graph = []

    # 创建资源节点和进程节点
    create_graph(resources, processes)

    # 运行资源分配算法
    result = Graph_Allocate_Algorithm()
    edges_output_container = []
    edges_output = []
    for graph in result:
        edges_output = []
        for edge in graph:
            edges_output.append({
                'source': edge.from_node,
                'target': edge.to_node
            })
        edges_output_container.append(edges_output)

    # 将结果转换为前端期望的格式
    output = {
        'nodes': [{'name': node.num, 'type': 'Resource' if isinstance(node, ResourceNode) else 'Process'} for node in
                  thread_local.Nodes],
        'edges': edges_output_container
    }

    return jsonify(output)

def dict_array_to_int_array(new):
    dict_array=new[:]
    if type(new[0])==dict:
        int_array = [item['count'] for item in dict_array]
        return int_array
    elif type(new[0])==int:
        print("here")
        return dict_array

def int_array_to_dict_array(new):
    int_array=new[:]
    dict_array = [{'name': chr(65 + i), 'count': count, 'exist': True} for i, count in enumerate(int_array)]
    return dict_array

def detect_deadlock(total_resources, processes):
    available = total_resources[:]
    available = dict_array_to_int_array(available)
    for process in processes:
        for i in range(len(total_resources)):
            available[i] -= process['allocation'][i]

    for process in processes:
        for i, req in enumerate(process['max_requirements']):
            if req > available[i]:
                return True
    return False


def rollback_to_resolve_deadlock(resources, processes):
    rollback_actions = []
    the_resources=resources[:]
    available = resources[:]
    available=dict_array_to_int_array(available)
    print(available)
    for process in processes:
        for i in range(len(the_resources)):
            available[i] -= process['allocation'][i]
    print(available)
    deadlock = detect_deadlock(the_resources, processes)
    while deadlock:
        for index, process in enumerate(processes):
            if any(process['allocation']):
                for i in range(len(resources)):
                    available[i] += process['allocation'][i]
                    rollback_actions.append(f"回滚进程{index+1}的资源{chr(i+ord('A'))}，从{process['allocation'][i]}减少到0")
                    process['max_requirements'][i]+=process['allocation'][i]
                    process['allocation'][i] = 0

                if not detect_deadlock(the_resources,processes):
                    return processes, rollback_actions
        print(processes)
        deadlock = detect_deadlock(the_resources, processes)

    return processes, rollback_actions


def cancel_to_resolve_deadlock(resources, processes):
    cancellation_actions = []
    available = resources[:]
    the_resources = resources[:]
    available = dict_array_to_int_array(available)
    for process in processes:
        for i in range(len(the_resources)):
            available[i] -= process['allocation'][i]
    print(available)
    deadlock = detect_deadlock(the_resources, processes)
    while deadlock:
        # 计算每个进程分配的资源总数
        process_totals = [sum(process['allocation']) for process in processes]
        # 找到资源总数最小的进程的索引
        process_index = process_totals.index(min(process_totals))
        # 撤销该进程
        process = processes.pop(process_index)
        for i in range(len(resources)):
            available[i] += process['allocation'][i]
            process['max_requirements'][i] += process['allocation'][i]
        cancellation_actions.append(f"撤销进程{process_index+2}")
        deadlock = detect_deadlock(the_resources, processes)
    return processes, cancellation_actions

@app.route('/allocate_resources_solve', methods=['POST'])
def allocate_resources_solve():
    # 从前端接收数据
    data = request.get_json()
    # print(type(data))
    resources = [resource for resource in data['resources']]
    # print(resources)
    processes = [process for process in data['processes']]

    thread_local.cnt1 = -1
    thread_local.cnt2 = -1
    thread_local.Nodes = []
    thread_local.Graph = []

    # 创建资源节点和进程节点
    create_graph(resources, processes)
    result=[]
    result.append(thread_local.Graph)
    Original_Nodes=thread_local.Nodes
    print(resources)
    print(processes)
    mark=0
    if data['mark']=='roll':
        mark=1
    elif data['mark']=='cancel':
        mark=0
    else:
        prints=["前端返回方法标记错误"]
        outputs={
            'nodes':None,
            'edges':None,
            'prints':prints
        }
        return jsonify(outputs)

    # print(resources)
    # 初始化线程局部变量
    thread_local.cnt1 = -1
    thread_local.cnt2 = -1
    thread_local.Nodes = []
    thread_local.Graph = []
    prints=[]
    if not detect_deadlock(resources,processes):
        prints="没有检测到死锁"
        outputs={
            'nodes':None,
            'edges':None,
            'prints':prints
        }
        return jsonify(outputs)
    if mark:
        processes,prints=rollback_to_resolve_deadlock(resources, processes)
    else:
        processes,prints=cancel_to_resolve_deadlock(resources, processes)
    # resources=int_array_to_dict_array(resources)
    # 创建资源节点和进程节点
    print(resources)
    print(processes)
    create_graph(resources, processes)

    # 运行资源分配算法
    result += Graph_Allocate_Algorithm()
    edges_output_container = []
    edges_output = []
    for graph in result:
        edges_output = []
        for edge in graph:
            edges_output.append({
                'source': edge.from_node,
                'target': edge.to_node
            })
        edges_output_container.append(edges_output)
    n=len(resources)
    m=len(processes)
    print(n)
    print(m)
    show_message = []
    msg = "系统开始使用 资源分配图算法 展示解决死锁后的进程调度！"
    show_message.append({"num": 1, "msg": msg})
    msg = f"创建资源{''.join(chr(x+ord('A')-1) + ',' for x in range(1, int(n/2)))}" + chr(int(n/2)+ord('A')-1)
    show_message.append({"num": 2, "msg": msg})
    msg = f"创建进程{''.join(str(x) + ',' for x in range(1, int(m/2)))}" + str(int(m/2))
    show_message.append({"num": 3, "msg": msg})

    # 将结果转换为前端期望的格式
    output = {
        'nodes': [{'name': node.num, 'type': 'Resource' if isinstance(node, ResourceNode) else 'Process'} for node in
                  Original_Nodes],
        'edges': edges_output_container,
        'prints': prints,
        'message' : show_message
    }
    print(prints)
    return jsonify(output)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002, debug=True)
