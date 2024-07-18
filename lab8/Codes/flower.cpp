# include <fstream>
# include <iostream>
# include <vector>
# include <sstream>
# include <cstdlib>
# include <omp.h>
# include <string.h>
# include <time.h>

# define INF 0x3f3f3f3f 
# define FILE_PATH "updated_flower.csv"
using namespace std;

struct Args{
    int source, target;     // 边的两个端点
    double distance;        // 两个端点之间的距离
};

int getRandomNum(int bottom, int top);      // 生成指定范围的整数
Args processLine(string line);      // 解析处理读取后的csv文件的每一行的节点和距离信息
void init(vector<Args>& messages, vector<vector<double>>& distances, int nodeNum);

// 根据对表格进行处理后得到的各节点之间的距离信息初始化距离矩阵
vector<double> dijkstra(vector<vector<double>> distances, int start, int thread_num);   // 使用Dijkstra算法来计算两个不同节点之间的最短距离

int main(){
    int MaxNode = 0;    // 记录数据集中的最大编号
    int MinNode = INF;  // 记录数据集中的最小编号
    ifstream fin(FILE_PATH);    // 用于读入表格    
    vector<Args> Messages;      // 存储表格信息
    int thread_num;     // 线程数
    cout << "请输入线程数：" << endl;
    cin >> thread_num;

    /*从给定的csv文件中读取信息*/ 
    if( !fin.is_open() ){
        cerr << "Failed to open the file" <<endl;
        exit(-1);
    }
    string line;    // 用于存储表格的每一行数据
    bool isFirstLine = true;    // 判断是否是标题行
    while (getline(fin, line)) {
            // 读取每一行
            if( isFirstLine ){      // 标题行不进行处理
                isFirstLine = false;
                continue;
            }
            Args Line = processLine(line);
            Messages.emplace_back(Line);
            MaxNode = max(Line.source, MaxNode);
            MaxNode = max(Line.target, MaxNode);
            MinNode = min(Line.source, MinNode);
            MinNode = min(Line.target, MinNode);
    }   


    /*使用Dijkstra算法前的准备工作*/ 
    int NodeNum = MaxNode + 1;      // 从0开始到MaxNode，共有（MaxNode+1）个节点
    vector<vector<double>> Distances;     // 用来存储节点之间的距离
    // vector<vector<int>> AdjacentNodes;  // 记录各个节点之间的邻接节点
    vector<bool> visited;
    init(Messages, Distances, NodeNum);     // 初始化距离矩阵

    /*使用Dijkstra算法计算节点之间的最短距离*/
    double start_time, using_time;        // 计时
    vector<vector<double>> shortestDistances;
    start_time = omp_get_wtime();

    for( int i = 0; i < NodeNum; ++i ){
        vector<double> dist = dijkstra(Distances, i, thread_num);
        shortestDistances.emplace_back(dist);
    }
    using_time = omp_get_wtime() - start_time;
    
    string test_data = "test_flower_data.csv";      // 随机生成的测试数据
    string test_result = "test_flower_result.csv";      // 测试数据的结果

    ofstream data(test_data);
    ofstream result(test_result);

    if (!data.is_open() || !result.is_open()) {
        cerr << "Failed to open the file for writing." << endl;
        exit(-1);
    }

    for( int i = 0; i < NodeNum; ++i ){
        int node_1 = getRandomNum(MinNode, MaxNode), node_2 = getRandomNum(MinNode, MaxNode);   // 随机生成2个节点
        data << to_string(node_1+1) <<"," << to_string(node_2+1)  <<endl;   // 写入测试数据
        result << to_string(node_1+1) <<"," << to_string(node_1+1) << "," << to_string(shortestDistances[node_1][node_2]) <<endl;   // 写入测试结果
        cout << "The shortest distance betwwen " << node_1 + 1 <<" and " << node_2 + 1 <<" is：" << shortestDistances[node_1][node_2] << endl;
    }

    cout<<FILE_PATH<<endl<<thread_num<<" threads using time："<<using_time<<" s"<<endl;
    
    // 关闭文件
    data.close();
    result.close();
    fin.close(); 
    return 0;
}

int getRandomNum(int bottom, int top){
    int random_num = rand() % (top - bottom + 1) + bottom;  //生成[bottom, top]范围内的随机整数
    return random_num;
}

Args processLine(string line){
    Args result;
    int comma_num = 0;   // 已经遇到的逗号数量
    int num_start_index = 0;    // 每个数字的起始索引
    for( int i = 0; i < line.size(); ++i ){
        if( line[i] == ','  && comma_num < 2){
            string numStr = line.substr(num_start_index, i - num_start_index);  // 读取数字部分的字符串
            int node = stoi(numStr)-1;    // 转换为整数，由于flower的节点ID是从1-930（区间内共930个整数），因此为防止出现段错误，需要-1，后面打印输出的时候+1即可
            if(comma_num == 0){     // 遇到第一个逗号，其前面是原来表格中的source项
                result.source = node;
            }
            else{       // 遇到第二个逗号，2个逗号之间是原来表格中的target项
                result.target = node;
            }
            num_start_index = i + 1;        // 更新下一个数字的起始索引
            comma_num++;        // 逗号数量加一
        }
        else{
            if (comma_num == 2 ){       // 已经遇到两个逗号，剩下的是distance项
                string numStr = line.substr(num_start_index, line.size() - num_start_index);    // 获取distance的数字字符串
                double dist = stod(numStr);     // 转为浮点数
                result.distance = dist;
                break;      
            }
        }
    }
    return result;
}

void init(vector<Args>& messages, vector<vector<double>>& distances, int nodeNum){
    distances.resize(nodeNum);
    for( int i = 0; i < nodeNum; ++i){
        distances[i].assign(nodeNum, INF);    // 先将所有距离全都初始化为无穷大
        distances[i][i] = 0;        // 节点与自身的距离为0
    }

    // 通过查阅messages中的节点之间的距离信息，修改对应的表项
    for( size_t i = 0; i < messages.size(); ++i ){
        int src = messages[i].source, tar = messages[i].target;
        double dist = messages[i].distance;
        distances[src][tar] = dist;
        distances[tar][src] = dist;
    }

    return;
}

vector<double> dijkstra(vector<vector<double>> distances, int start, int thread_num){
    int nodeNum = distances.size();
    vector<bool> visited;
    visited.assign(nodeNum, false);
    visited[start] = true;
    vector<double> dist = distances[start];

    for( int i = 0; i < nodeNum; ++i ){
        int min_dist, min_dist_index;
        min_dist = INF;

        # pragma omp parallel for num_threads(thread_num) shared(dist, visited)
        for( int j = 0; j < nodeNum; ++j){
            if( !visited[j] && dist[j] < min_dist){
                min_dist = dist[j];
                min_dist_index = j;
            }
        }
        

        visited[min_dist_index] = true;
        # pragma omp parallel for num_threads(thread_num) shared(dist, visited)
        for( int k = 0; k < nodeNum; ++k ){
            if( !visited[k] && dist[k] > dist[min_dist_index] + distances[min_dist_index][k]){
                dist[k] = dist[min_dist_index] + distances[min_dist_index][k];
            }
        }
    }
    return dist;
}