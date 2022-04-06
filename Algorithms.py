#                          BFS
 
	# Vertices is a 2D list defining edges for each vertex
	# Here Visited array = vis  --- It also stores min distance of that vertex from root

from collections import deque
def bfs(vertices, start, dest):      
    q = deque()
    q.append(start)
    vis = [0]*(n+1)          
    while q:
        u = q.popleft()                          
        for v in vertices[u]:
            if vis[v]==0 and v!=start:
                vis[v] = vis[u] + 1
                q.append(v)
    return vis[dest]
 
 
 
    #     DFS to count number of nodes 
 
 
def dfs(x,k=1):
	vis[x]=1
	for v in g[x]:
		if vis[v]==0:
			k=1+dfs(v,k)
	return k
 
 
 
 
 # Post Order Traversal DFS for finding subtree
 
 def dfs(x):
	vis[x]=1
	t=[x]
	for i in v[x]:
		if vis[i]==0:
			t.extend(dfs(i))
		#	print(t)
	z[x]=t             # Base case
	return t
v=[[],[2,5],[1,3,4],[2,6],[2,7,8],[1],[3],[4],[4]]
vis=[0]*9
z=[0]*9
dfs(1)
print(z)
 
 
#             Disjoint Set Union
 
 
def getP(x):        #  Get parent with Path Compression
    while x!=p[x]:
        x=p[x]
        p[x]=p[p[x]]     
    return x
 
def union(u,v):         
    p1,p2=getP(u),getP(v)
    if p1!=p2:
        p[p1]=p2
 
 
size=[1]*n
def union(a,b):     # Union by size/rank
    rootA = getP(a)
    rootB = getP(b)
    if rootA == rootB:
        return
    if size[rootA]<size[rootB]:
        arr[rootA] = arr[rootB]
        size[rootB] = size[rootB] + size[rootA]
    else:
        arr[rootB] = arr[rootA]
        size[rootA] = size[rootA] + size[rootB]
        
        
        
        
#                Djikstra's Algorithm
#       For finding path between vertex 1 and n

from heapq import *       
g=[[] for i in range(n+1)]
for i in range(m):
	x,y,w=map(int,input().split())
	if x!=y:
		g[x].append((y,w))
		g[y].append((x,w))
path=[i for i in range(n+1)]
dis=[10**12]*(n+1)
hp=[(0,1)]
while hp:
	dcur,cur=heappop(hp)
	for v,w in g[cur]:
		if dcur+w<dis[v]:
			dis[v]=dcur+w
			path[v]=cur
			heappush(hp,(dis[v],v))
l=[n]
x=n
if dis[n]!=10**12:
	while x!=1:
		x=path[x]
		l.append(x)
	print(*l[::-1])
else:print(-1)


##                   Linked List Template
class Node:
	def __init__(self,data):
		self.data=data
		self.next=None
class LList:
	def __init__(self):
		self.head=None
	def push(self,newdata):
		newnode=Node(newdata)
		newnode.next=self.head
		self.head=newnode
	def printlist(self):
		t=self.head
		while t:
			print(t.data, end=' ')
			t=t.next
l=LList()
l.push(1)
l.push(2)
l.push(3)
l.push(4)
l.printlist()


#               LCA using Binary Lifting and sparse table


import math
# To calculate level of each node
def dfs(x,prev):
	lvl[x]=lvl[prev]+1
	for u in g[x]:
		if u!=prev:
			dfs(u,x)
			
# To calculate lca of u and v in logN
def LCA(u,v):
	if lvl[u]>lvl[v]:
		u,v=v,u
	dif=lvl[v]-lvl[u]

	for i in range(maxn+1):    #    while dif>0: 
		if (dif>>i)&1:         #        i=int(math.log2(dif))    
			v=lca[v][i]        #        v=lca[v][i]
                               #        d-=(1<<i)
	# now u and v are at same level
	if u==v: 
		return u

	for i in range(maxn,-1,-1):
		if lca[v][i]!=lca[u][i]:
			u=lca[u][i]
			v=lca[v][i]

	return lca[u][0]
	
	
n,q=map(int,input().split())
par=[-1,-1]+list(map(int,input().split()))
lvl=[-1]*(n+1)
lca=[[]]
g=[[] for i in range(n+1)]
for i in range(2,n+1):
	g[i].append(par[i])
	g[par[i]].append(i)

dfs(1,0)

maxn=int(math.log2(max(lvl)))  # max power of 2(column of sparse table)
lca=[[-1]*(maxn+1) for i in range(n+1)]   # lca[i][j] stores 2^j th parent of i th node

# for 2^0 th parent(base case)
for i in range(1,n+1):
	lca[i][0]=par[i]

# dp method for rest 2^j
for j in range(1,maxn+1):
	for i in range(n+1):
		if lca[i][j-1]!=-1:
			lca[i][j]=lca[lca[i][j-1]][j-1]

for i in range(q):
	u,v=map(int,input().split())
	ans=LCA(u,v)
	if ans==-1:
		print(1)
	else:print(ans)



####                                   TRIE

class TrieNode:
    # Initialize your data structure here.
    def __init__(self):
        self.word=False
        self.children={}

class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node=self.root
        for i in word:
            if i not in node.children:
                node.children[i]=TrieNode()
            node=node.children[i]
        node.word=True

    def search(self, word):
        node=self.root
        for i in word:
            if i not in node.children:
                return False
            node=node.children[i]
        return node.word

    def startsWith(self, prefix):
        node=self.root
        for i in prefix:
            if i not in node.children:
                return False
            node=node.children[i]
        return True
