#动态规划法试图仅仅解决每个子问题一次，具有天然剪枝的功能，从而减少计算量
#一旦某个给定子问题的解已经算出，则将其记忆化存储（多少状态多大的表）
#以便下次需要同一个子问题解之时直接查表。
#这种做法在重复子问题的数目关于输入的规模呈指数增长时特别有用。

#dp:自底向上的解决问题。重叠子问题，最优子结构。
#没有思路的时候先想暴力解法，dp往往需要先画一个dp树的特别是非契波题的时候。
#先分析清楚自顶向下用暴力回溯递归的思路后，再想怎么从底向上。
#状态与状态的转移问题。对回溯的记忆存储问题。



#300，求最长上升子序列长度
#动态规划求解
def lengthOfLIS():
     nums=[10,9,2,5,3,4]

     if len(nums)==0:
          return 0
     dp=[0 for _ in range(len(nums))]#dp状态记录i比多少数大
     dp[0]=1
     m=1
     for i in range(len(nums)):
          maxval=0
          for j in range(0,i):#对每个i都往前面找
               if nums[i]>nums[j]:
                    maxval=max(maxval,dp[j])#寻找比自己小的最大长度
          dp[i]=maxval+1 #再算上自己，构成更长的长度
          m=max(m,dp[i]) #更新最大值返回
     return m

#递归容易超出时间限制
def lengthOfLIS1():
     nums=[10,9,2,5,3,4]

     if len(nums)==0:
          return 0

     def lis(pre,ind,count): #主要是练手
          if ind==len(nums):
               return count
          cur=nums[ind]
          print(pre,cur,count)
          a,b=0,0
          if cur>pre: #用当前节点
               b=lis(pre,ind+1,count)
               count+=1
               a=lis(cur,ind+1,count)
          else: #不用当前节点
               b=lis(pre,ind+1,count)

          return max(a,b) #用与不用返回最大值
     return lis(-1000000000,0,0)
#贪心+二分查找优化，优化了10倍不止啊....
#思路是维护一个贪心的列表，记录最小值，查询用二分就logn了，整个就降为nlogn。
def lengthOfLIS2():
     nums=[10,9,2,5,3,4]

     if len(nums)==0:
          return 0

     tails,res=[0 for _ in range(len(nums))],0
     for num in nums:
          i,j=0,res
          while i<j:
               m=(i+j)//2
               if tails[m]<num:#如果有相同数就加个等于，让i往后移
                    i=m+1
               else:
                    j=m
          tails[i]=num #i即是可插入的最小处
          if j==res: #记录tails的已有元素的大小
               res+=1
     return res
#print(lengthOfLIS2())


#最大最小最重最长--优化问题，可以用递归，但是有时候会出现重叠子问题，如切波那契会重复算一些值，O2^n
#需要优化，需要记住状态，所以dynamical programming是高效穷举算法。递归+记忆+猜测


#DP法，字符串匹配很适合用dp解呀。
def isMatch():
     s="acdcb"
     p="b*c?b"
     s="aa"
     p="*"
     
     n, m = len(s), len(p)
     dp=[[False for _ in range(m+1)] for _ in range(n+1)]#建一个dp表，+1是存在空串

     #初始化
     dp[0][0]=True #都空则为True
     #初始化
     for j in range(1,m+1):
          if p[j-1]=="*":
               dp[0][j]=dp[0][j-1] and p[j-1]=="*" #s空，p需要为连续的*才行，不然前面就曾经已经断过了。

     #填表
     for i in range(1,n+1):
          for j in range(1,m+1):
               if p[j-1]==s[i-1] or p[j-1]=="?": #-1也是因为表多+1，所以找p，s要-
                    dp[i][j]=dp[i-1][j-1] #此时匹配需要看前面是否为True
               elif p[j-1]=="*":
                    dp[i][j]=dp[i][j-1] or dp[i-1][j] #*需要看匹配一个或者多个

     print(dp)
     return dp[n][m] #得到表尾

#不能取相邻数以得到最大值
#状态只有2种，似乎不用填dp表，之间往后走就行。。
def massage():
     nums=[1,2,3,1]

     n=len(nums)
     if n==0:
          return 0
     dp0,dp1=0,nums[0] #不选0号，选0号

     for i in range(1,n): #遍历到列表尾
          tdp0=max(dp0,dp1)#0表示不选当前位，所以可以得到前一位的两者最大
          tdp1=dp0+nums[i]#选当前位，所以只能不选前一位，故是dp0+自己的数
          #更新
          dp0=tdp0
          dp1=tdp1

     return max(dp0,dp1)
#print(massage())


#连续最大和
def maxSubArray():
     nums=[-2,1,-3,4,-1,2,1,-5,4]

     if len(nums)==0: return 0

     dp=[0 for _ in range(len(nums))] #先建表，再初始化
     dp[0]=nums[0]
     res=nums[0]

     for i in range(1,len(nums)):
          dp[i]=max(dp[i-1]+nums[i],nums[i]) #max与前一项连续 vs 不连续就只有自己
          res=max(res,dp[i]) #保存最大的连续和最大

     return res
#print(maxSubArray())

#爬楼梯可抽象成契波。
def climbStairs():
     n=10

     if n==1 or n==2:return n
     
     dp=[0 for _ in range(n)]
     dp[0],dp[1]=1,2
     for i in range(2,n):
          dp[i]=dp[i-1]+dp[i-2]#一步或者两步
     print(dp)
     return dp[-1]

def climbStairs():
     n=10

     if n==1 or n==2:return n

     a,b=1,2
     for i in range(2,n):
          a,b=b,a+b #只有两种状态，所以可以用常量代替
     return b
#print(climbStairs())



def minPathSum():
     grid=[[1,3,1],
           [1,5,1],
           [4,2,1]]

     dp=[[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]
     dp[0][0]=grid[0][0] #建表初始化
     for i in range(1,len(grid)):
          dp[i][0]=grid[i][0]+dp[i-1][0]
     for j in range(1,len(grid[0])):
          dp[0][j]=grid[0][j]+dp[0][j-1]
     for i in range(1,len(grid)):
          for j in range(1,len(grid[0])):
               dp[i][j]=min(dp[i-1][j],dp[i][j-1])+grid[i][j]#每次都是从左边来的，或者从上边来的点
     return dp[-1][-1] #到达终点，最小的消耗是最后的值

#print(minPathSum())



#62,迷宫左上到右下，不同路径的数目
#故意先写了个回溯，不出意料超时。在数目都变成10+的时候就超了，注意测试数据的大小。
def uniquePaths():
     m,n=3,7

     res=[0]
     def back(i,j):
          print(i,j)
          if i==m-1 and j==n-1:
               res[0]+=1

          if i<0 or j<0 or i>=m or j>=n:
               return

          back(i+1,j)
          back(i,j+1)
     back(0,0)
     return res[0]

#果然应该用DP
def uniquePaths():
     m,n=8,7

     dp=[[0 for _ in range(m)] for _ in range(n)]
     dp[0][0]=1
     for i in range(1,m):
          dp[0][i]=dp[0][i-1]
     for j in range(1,n):
          dp[j][0]=dp[j-1][0]

     #这个建表+初始化好强啊，速度和空间都优化了。
     #dp = [[1]*m]+[[1]+[0]*(m-1) for _ in range(n-1)] #只有a=b是浅复制似乎，矩阵的*没问题。
     #print(dp)
     
     for i in range(1,n):
          for j in range(1,m):
               dp[i][j]=dp[i-1][j]+dp[i][j-1]#从左来或者从上来
     return dp[-1][-1]    
#print(uniquePaths())

#63,比62多了一个障碍物怎么办
def uniquePathsWithObstacles():
     obstacleGrid=[[0,0,0],
                   [0,1,0],
                   [0,0,0]]
     obstacleGrid=[[1]]

     if obstacleGrid[0][0]==0:#一开始就为障碍的特殊情况初始化
          obstacleGrid[0][0]=1
     else: return 0
     
     for i in range(1,len(obstacleGrid[0])):
          if obstacleGrid[0][i]==0:
               obstacleGrid[0][i]=obstacleGrid[0][i-1]
          else: obstacleGrid[0][i]=0 #有障碍就直接为0
     for j in range(1,len(obstacleGrid)):
          if obstacleGrid[j][0]==0:
               obstacleGrid[j][0]=obstacleGrid[j-1][0]
          else: obstacleGrid[j][0]=0
          
     for i in range(1,len(obstacleGrid)):
          for j in range(1,len(obstacleGrid[0])):
               if obstacleGrid[i][j]==1:#有障碍就直接为0
                    obstacleGrid[i][j]=0
               #不然就是从左变的数+上面来的数
               else: obstacleGrid[i][j]=obstacleGrid[i-1][j]+obstacleGrid[i][j-1]
     return obstacleGrid[-1][-1]
#print(uniquePathsWithObstacles())                 


#91,对数字合法的分割
#回溯毫无疑问超时
def numDecodings():
     s="120041"

     l=len(s)
     res=[0]
     def back(i,tmp):
          print(i,tmp)
          if int(tmp)==0 or int(tmp)>26 or len(str(int(tmp)))!=len(tmp):
               return
          if i==l:
               res[0]+=1

          if i<l:
               back(i+1,s[i:i+1])
          if i<l-1:
               back(i+2,s[i:i+2])
     back(0,s[0])
     return res[0]

def numDecodings():
     s="120451114241"

     dp=[0 for _ in range(len(s))]
     #是否合法。1数在范围中 2首位不为0
     def Isture(tmp):
          return int(int(tmp)>0 and int(tmp)<=26 and len(str(int(tmp)))==len(tmp))

     if len(s)==1:
          return Isture(s)
     if len(s)==2:
          return Isture(s)+(Isture(s[0]) and Isture(s[1]))

     dp[0]=Isture(s[0])
     dp[1]=Isture(s[0:2])+(Isture(s[0]) and Isture(s[1]))
     for i in range(2,len(s)):
          print(dp)
          if s[i]=="0":#当前为0，说明自己不能单独存在和作为首项
               if Isture(s[i-1:i+1]):#只能检查看能否与前一位构成10，20
                    dp[i]=dp[i-2] #如果能，0将绑架i-1，将只能等于dp[i-2]
               else:
                    return 0#如果不能就违法直接没有划分了
          else:#不为0的数字一定能自己单独存在的
               print(s[i-1:i+1],Isture(s[i-1:i+1]))
               if Isture(s[i-1:i+1]):#如果能和前一项构成合法数
                    dp[i]=dp[i-1]+dp[i-2] #等于选择单独自己i-1，绑架前一位i-2
               else:
                    dp[i]=dp[i-1]#不然就只能自己i-1
     return dp[-1]
#print(numDecodings())


#求最长连续合理括号。
def longestValidParentheses():
     s="()((()))()())))()()())))"

     dp=[0 for _ in range(len(s))]
     for i in range(1,len(s)):
          if s[i]==")":#判断合理与否是）出现定论的，所以（们都是0
               if s[i-1]=="(":#如果前一个就是（
                    if i-2>=0:
                         dp[i]=dp[i-2]+2 #加2的同时，应该判断是否与前面的相连
                    else:
                         dp[i]=2
               else: #如果前一个不是（，需要判断是不是嵌套的括号
                    if i-dp[i-1]>0 and s[i-1-dp[i-1]]=="(": #所以要减去i-1那个地方的dp[i-1]（它前面的连续括号长度）找它前面的地方是不是（
                         if i-2-dp[i-1]>=0: #是说明这一段是连续的，然后需要再加上跟（前面是否是连续的
                              dp[i]=dp[i-1]+dp[i-2-dp[i-1]]+2
                         else:
                              dp[i]=dp[i-1]+2

     print(dp)
     return max(dp)
#暴力解，对每个偶数个字符都做是否符合的判断。range(0,n,1),range(i,n,2)，再记住m。
def isValid(s):
     stack=[] #括号用栈天经地义
     for i in range(len(s)):
          if s[i]=="(":
               stack.append("(")
          elif len(stack)>0 and stack[-1]==")":
               stack.pop()
          else: #不匹配就F
               return False
     return len(stack)>0 #只有有人多，也是F
#栈解
def longestValidParentheses():
     s="()((()))()())))()()())))"

     stack,m=[-1],0 #-1是哨兵，保证每个括号都能被比较
     for i in range(len(s)):
          if s[i]=="(": #左括号就进栈
               stack.append(i)
          else:
               stack.pop() #右括号就先pop
               if len(stack)==0: #pop完是空，表示）多了，不连续
                    stack.append(i) #压入i做哨兵
               else: #不是空说明连续
                    m=max(m,i-stack[-1])#计算与栈顶下标的差距，会一直计算一直到空所以很完备的
     return m
#虚拟栈指针模拟。
def longestValidParentheses():
     l,r,m=0,0,0
     for i in range(len(s)):
          if s[i]=="(":
               l+=1
          else:
               r+=1

          if l==r:#只要相等了，就说明合理！！
               m=max(m,2*r)
          elif r>l:#）括号只要敢多，就断了！！所以直接为0
               r,l=0,0

     l,r==0
     for i in range(len(s)-1,-1,-1):
          pass
#print(longestValidParentheses())


     
#求自顶向下的最小路径。
def minimumTotal():
     triangle=[[2],
               [3,4],
               [6,5,7],
               [4,1,8,3]]

     for i in range(1,len(triangle)):
          for j in range(i+1):
               if j==0:#两个边界都是只有一条路可以到达的
                    triangle[i][j]+=triangle[i-1][0]
               elif j==i:
                    triangle[i][j]+=triangle[i-1][j-1]
               else:#中间的有两条路
                    triangle[i][j]+=min(triangle[i-1][j],triangle[i-1][j-1])
     return min(triangle[-1])
#自底向上！！！
def minimumTotal():
     for row in range(len(triangle)-2,-1,-1):#从倒数第二行开始
          for col in range(len(triangle[row])):
               #自底向上没有边界问题，一定都是两条边。
               triangle[row][col]+=min(triangle[row+1][col],triangle[row+1][col+1])
     return triangle[0][0]
#print(minimumTotal())




#股票套题......
#交易一次
def maxProfit():
     prices=[7,2,5,8,3]#k线数组
     if len(prices)==0:
          return 0
     dp=[0 for _ in range(len(prices))]
     b=0#只能交易一次找最小和向右的最大
     for i in range(1,len(prices)):
          if prices[i]<=prices[i-1]:#比前一个还小就不用算了
               dp[i]=dp[i-1]
          else:
               if prices[i-1]<prices[b]:
                    b=i-1 #更新最小
               dp[i]=max(prices[i]-prices[b],dp[i-1])#比较在b卖出的钱和不卖继续向后找
     return dp[-1]


#交易无限制
def maxProfit():
     prices=[7,1,5,3,6,4]
     prices=[7,6,4,3,1]

     prices+=[-1]
     i,j,count=0,1,0
     while j<len(prices):
          if prices[j-1]>=prices[j]:#既然无限制，找到一个上升的就做操作
               count+=prices[j-1]-prices[i]
               i=j
               j+=1
          else:
               j+=1
     return count
def maxProfit():
     profit=0
     for i in range(1,len(prices)):#更直接，对每一小段都算一遍....
          tmp=prices[i]-prices[i-1]
          if tmp>0: profit+=tmp
     return profit
def maxProfit():
     l=len(prices)
     if l<2: return 0
     dp=[[0 for _ in range(2)] for _ in range(l)] #dp法。0表示不持有，1表示此时持有
     dp[0][0]=0
     dp[0][1]=-prices[0] #持有相当于买进，所以减去价格

     for i in range(1,l):
          dp[i][0]=max(dp[i-1][0],dp[i-1][1]+prices[i]) #不持有。可能以前就卖了，或者今天卖的。
          dp[i][1]=max(dp[i-1][1],dp[i-1][0]-prices[i]) #持有。可能以前买的，或者今天买的。
     return dp[-1][0]


#只能交易2次
def maxProfit():
     #prices=[3,3,5,0,0,3,1,4]
     prices=[3,3,5,0,0,3,1,4,5,32,12,5,21,36]

     l=len(prices)
     if l<2: return 0
     dp=[[[0 for _ in range(3)] for _ in range(2)] for _ in range(l)]
     #dp法此时再加一个变量，标识已经卖了几次了，最多2次，状态是0，1，2
     #不要怕麻烦好吗？他该3个dp就3dp怎么了。
     dp[0][0][0]=0
     dp[0][0][1]=float('-inf')#0时刻不可能卖过，所以4个状态都是-inf
     dp[0][0][2]=float('-inf')
     dp[0][1][0]=-prices[0]
     dp[0][1][1]=float('-inf')
     dp[0][1][2]=float('-inf')

     for i in range(1,l):
          dp[i][0][0]=0 #不持有，没卖过。啥也没干就是0.
          dp[i][0][1]=max(dp[i-1][1][0]+prices[i],dp[i-1][0][1])#不持有，卖过。今天卖或者以前就卖了。
          dp[i][0][2]=max(dp[i-1][1][1]+prices[i],dp[i-1][0][2])
          dp[i][1][0]=max(dp[i-1][0][0]-prices[i],dp[i-1][1][0])#持有，没卖过。今天买或者以前买。
          dp[i][1][1]=max(dp[i-1][0][1]-prices[i],dp[i-1][1][1])
          dp[i][1][2]=float('-inf')#卖了2次之后不能再持有了。
     return max(dp[-1][0])

#只能交易k次
def maxProfit():
     prices=[3,3,5,0,0,3,1,4,5,32,12,5,21,36]
     k=8
      
     l=len(prices)
     if l<2: return 0
     #直接套用上题会超时。细想每次买卖都需要2天，如果k大于了l//2
     #那就说明没限制了，直接回到第2题算每段。
     if k>l//2:
          profit=0
          for i in range(1,len(prices)):
               tmp=prices[i]-prices[i-1]
          if tmp>0: profit+=tmp
          return profit

     #不然就改装第3题。状态是k+1个。
     dp=[[[0 for _ in range(k+1)] for _ in range(2)] for _ in range(l)]
     dp[0][0][0]=0
     dp[0][1][0]=-prices[0]
     for t in range(1,k+1):
         dp[0][0][t]=float('-inf')
         dp[0][1][t]=float('-inf')

     for i in range(1,l):
          for t in range(k+1):#用for填表，分三种情况。
               if t==0:
                    dp[i][0][0]=0
                    dp[i][1][0]=max(dp[i-1][0][0]-prices[i],dp[i-1][1][0])
               elif t==k:
                    dp[i][0][t]=max(dp[i-1][1][t-1]+prices[i],dp[i-1][0][t])
                    dp[i][1][t]=float('-inf')
               else:
                    dp[i][0][t]=max(dp[i-1][1][t-1]+prices[i],dp[i-1][0][t])
                    dp[i][1][t]=max(dp[i-1][0][t]-prices[i],dp[i-1][1][t])

     return max(dp[-1][0])
#print(maxProfit())


#打家劫舍3套题......
#普通版本，不能打劫相领的房子
def rob():
     nums=[2,7,9,3,1]
     nums=[1]

     if len(nums)==0:
          return 0
     dp=[0 for _ in range(len(nums))]
     for i in range(len(nums)):
          if i==0:
               dp[i]=nums[i]
          elif i==1:
               dp[i]=max(nums[i],nums[i-1])
          else:
               dp[i]=max(dp[i-2]+nums[i],dp[i-1])#当前+前2或者前一个。
     return dp[-1]
#print(rob())


#房间成环。首尾房间不能同时被抢，只可能有三种不同情况：
#1要么都不被抢2要么第一间房子被抢最后一间不抢3要么最后一间房子被抢第一间不抢。
def rob():
     if len(nums)==0:
          return 0
     #1不划算。所以考虑后两种，那就把上一题的代码在有头或者有尾算两次就得了。。
     return max(rob_one(nums[:-1]),rob_one(nums[1:])) if len(nums) != 1 else nums[0]

#房间变成了二叉树。
#二叉树可能首先要搜索，从底向上是个很好的想法。
class TreeNode:
     def __init__(self, x):
          self.val = x
          self.left = None
          self.right = None
def rob():
     root=[3,2,3,null,3,null,1]

     def robinteger(root):
          res=[0,0]#选这个节点或者不选
          if not root: return res
          left=robinteger(root.left)
          right=robinteger(root.right)#先到树的最末端
          res[1]+=root.val+left[0]+right[0] #选该父节点，那么孩子节点绝对不能选
          res[0]+=max(left[0],left[1])+max(right[0],right[1])#不选父节点的话，两个子树都能选，用最大的值加起来
          return res
     res=robinteger(root)
     return max(res[0],res[1])
#print(rob())



#找在字典中存在的单词划分。
#递归写法必然超时，存在重复子问题即重复出现的词。
def wordBreak():
     s = "abcd"
     wordDict = ["a","abc","b","cd"]

     l=len(s)
     wordset=set(wordDict)#变set加速
     def back(k):
          if k==l:
               return True
          for i in range(k+1,l+1):#自己存在，判断自己向后的字母是否存在
               if (s[k:i] in wordset) and back(i):
                    return True
          return False
     return back(0)
#print(wordBreak())
#dp法
def wordBreak():
     s = "leetcode"
     wordDict = ["leet","code"]

     wordset=set(wordDict)
     dp=[0 for _ in range(len(s)+1)]
     dp[0]=1
     #不要怕循环多
     for i in range(1,len(s)+1):#对每个划分点填表
          for j in range(i):#将划分点之前的字符串判断是否存在
               if dp[j] and s[j:i] in wordset: #即dp[j]存在，那么向后直到划分点是否存在
                    dp[i]=1
                    break
     print(dp)
     return bool(dp[-1])
#不仅要判断存在，返回所有的划分结果。
def wordBreak():
     s = "abcd"
     wordDict = ["a","abc","b","cd","d"]

     wordset=set(wordDict)
     dp=[[] for _ in range(len(s)+1)]#直接用dp存

     dp[0].append("")#有了东西，len(dp)就大于0了，等同之间的状态判断
     for i in range(1,len(s)+1):
          print(i,dp)
          li=[]#存下每词向前所有的可能
          for j in range(i):
               if len(dp[j])>0 and s[j:i] in wordset:
                    #此时dp的地方是满足的，dp后的地方也满足
                    #对于dp里面的所有结果都加上此时的值就行
                    for l in dp[j]:
                         if l=="":#特殊处理dp[0]
                              li.append(l+s[j:i])
                         else:
                              li.append(l+" "+s[j:i])
          dp[i]=li#这就是i的状态！
     return dp[-1]
#print(wordBreak())



#343,分成两个数并使数的乘积最大。
#至少分割成两个部分
def integerBreak():
     n=10

     if n==1: return 1 #1是无法分割的
     dp=[0 for _ in range(n+1)]
     dp[1],dp[2]=1,1
     for i in range(3,n+1):
          li=[]
          for j in range(1,i):
               tmp=i-j #每次都分割成j，i-j
               li.append(max(dp[tmp],tmp)*j)#直接两部分，或者对i-j继续分割
          dp[i]=max(li)
     return dp[-1]
#print(integerBreak())

#279，能分成平方数之和的最小个数
def numSquares():
     n=4

     dp=[0 for _ in range(n+1)]
     dp[1]=1
     squ=[z**2 for z in range(1,n) if z**2<=n] #平方和数组
     for i in range(2,n+1):
          count=float('inf')
          for j in squ:
               if i<j:break
               count=min(count,dp[i-j]+1)#不用这个平方数，和用这个平方数
          dp[i]=count
     return dp[-1]
#print(numSquares())




#0-1背包问题！！
def bag01():
     w=[1,2,3,4]
     v=[2,5,3,4]
     bag=5

     dp=[[0 for _ in range(bag+1)] for _ in range(len(w))]
     for i in range(bag+1):
          if w[0]<=i:
               dp[0][i]=v[0]
     print(dp)
     for i in range(1,len(w)):
          for j in range(1,bag+1):
               dp[i][j]=dp[i-1][j]#先不方放自己，看前一个的结果
               #两行分奇数偶数存
               #dp[i%2][j]=dp[(i-1)%2][j]
               if j-w[i]>=0:
                    dp[i][j]=max(dp[i][j],dp[i-1][j-w[i]]+v[i]) #能放自己就放入求max
     print(dp)
     return dp[-1][-1]
#print(bag01())
#对空间复杂度的优化。每次更新只依赖前一行，所以可以只保持两行的数据。
#一行就够啦！！
#for i in range(1,len(w)):
#    for j in range(bag,w[i]-1,-1):#必须是逆序防止这一行的结果已经被改动
#         dp[j]=max(dp[j],v[i]+dp[j-w[i]])

#背包问题的变体：
#完全背包问题，再加一个for对满足重量的范围，可以无限次放入i，每次都比较当前和再减自己之后的值谁最大就可以了。
#继续加限制。除了重量，对件数呀之类的也限制，此时相当于多了一个状态而已，变三维dp
#物品之间的互斥和依赖关系。把状态方程变一变。


#416，分成两个列表（背包）中的sum（重量）相等
def canPartition():
     nums=[1,5,11,5,6]

     if sum(nums)%2==0:#必须能等分，然后两个背包各装一半
          bag=sum(nums)//2
     else:
          return False

     dp=[[0 for _ in range(bag+1)] for _ in range(len(nums))]
     for j in range(bag+1):
          if nums[0]<=j:
               dp[0][j]=nums[0]
     for i in range(1,len(nums)):
          for j in range(1,bag+1):
               dp[i][j]=dp[i-1][j]
               if nums[i]<=j:
                    dp[i][j]=max(dp[i][j],dp[i-1][j-nums[i]]+nums[i])
          if dp[i][-1]==bag:
               return True
     return False
def canPartition():
     nums=[1,5,11,5]

     if sum(nums)%2==0:
          bag=sum(nums)//2
     else:
          return False
     dp=[False for _ in range(bag+1)]#以价值做一维dp就可以了
     for j in range(bag+1):
          dp[j]=(nums[0]==j)
     for i in range(1,len(nums)):
          for j in range(bag,nums[i]-1,-1):#必须逆序！
               dp[j]=dp[j] or dp[j-nums[i]]
     return dp[-1]
#print(canPartition())


#322，可重复的硬币组成某和的最小枚数
def coinChange():
     coins=[1,2,5]
     amount=11

     dp=[float('inf') for _ in range(amount+1)]#以和做dp维度
     for i in range(0,amount+1,coins[0]):
          dp[i]= i//coins[0]

     for i in range(1,len(coins)):
          #for j in range(amount,coins[i]-1,-1):
          for j in range(coins[i],amount+1):#必须正序！！因为可重复使用，就需要它不断的+
               print(j,dp)
               dp[j]=min(dp[j],dp[j-coins[i]]+1)

     if dp[-1]==float('inf'):
          return -1
     else: return dp[-1]
     #return dp[-1] if dp[-1]!=float('inf') else -1

#print(coinChange())

#377，组成某和的所有可能数目，考虑其顺序！
#没有很好的思路先写回溯
def combinationSum4():
     nums = [1, 2, 3,4,5]
     target = 10

     res=[0]
     def back(t):
          if t==target:
               res[0]+=1
               return
          for n in nums:
               if t+n<=target:
                    back(t+n)
          return

     back(0)
     return res[0]
#dp法
def combinationSum4():
     nums = [1, 2, 3]
     target = 4

     if len(nums)==0: return 0
     dp=[0]*(target+1)#一维的数组可以这样初始化的，多维就不行了
     dp[0]=1
     for j in range(1,len(dp)):#考虑顺序所以每种value都有多种组合
          for i in range(len(nums)):#然后再对组合进行for
               if j>=nums[i]:
                    dp[j]=dp[j]+dp[j-nums[i]]
     print(dp)
     return dp[-1]
#print(combinationSum4())

#1.如果是0-1背包，即数组中的元素不可重复使用，nums放在外循环，target在内循环，且内循环倒序；
#2.如果是完全背包，即数组中的元素可重复使用，nums放在外循环，target在内循环。且内循环正序。
#3.如果组合问题需考虑元素之间的顺序，需将target放在外循环，将nums放在内循环。




#474一和零
def findMaxForm():
     strs={"10", "0001", "111001", "1", "0"}
     m = 5 #0
     n = 3 #1

     dp=[[0 for _ in range(m+1)] for _ in range(n+1)]#存0和1的状态
     arr=[]
     #对原strs进行处理，明确知道它有几个1，0来组成
     for st in strs:
          i,j=0,0
          for s in st:
               if s=="1":i+=1
               else: j+=1
          arr.append([i,j])

     for s in range(len(strs)):#组成了一个就不用再组成了，所以是不重复背包问题
          for i in range(n,-1,-1):
               if i<arr[s][0]: continue
               for j in range(m,-1,-1):                
                    if j<arr[s][1]: continue
                    dp[i][j]=dp[i-arr[s][0]][j-arr[s][1]]+1  
     print(dp)
     return dp[-1][-1]
#print(findMaxForm())


#494目标和，给数组算+，-公式等于S。
#题目状态分析本身不难，但是！有+有-之后，搜索范围就扩大了。
def findTargetSumWays():
     nums=[1, 1, 1, 1, 1]
     S=3

     if S>1000: return 0 #题目给的最大数是1000

     dp=[[0 for _ in range(2001)] for _ in range(len(nums))]#仍然还是以value做dp
     dp[0][nums[0]+1000]=1
     dp[0][-nums[0]+1000]+=1 #用+=可以处理0这样就会是2的结果

     for i in range(1,len(nums)):#按元素来一个一个添符号
          for j in range(-1000,1001):#值的范围
               if dp[i-1][j+1000]>0:#因为范围扩大之后，原先的i会变成在数组中的i+1000
                    #当前的值可+可-，都是在前一个结果的基础上
                    dp[i][j+nums[i]+1000]+=dp[i-1][j+1000]
                    dp[i][j-nums[i]+1000]+=dp[i-1][j+1000]
     #print(dp)
     return dp[-1][S+1000]#S也变成在数组中的+1000的位置上         
#print(findTargetSumWays())




#376，求最长的波动型序列
def wiggleMaxLength():
     nums=[1,2,3,4,5,6,7,8,9]

     if len(nums)==0: return 0
     dp=[0 for _ in range(len(nums))]#用+，-来指示现在是up的还是down
     dp[0]=1
     for i in range(1,len(nums)):
          if nums[i]>nums[0]:
               dp[i]=2#不能构成波的就是数个数本身
          elif nums[i]<nums[0]:
               dp[i]=-2
          else: dp[i]=0

     for i in range(1,len(nums)):
          for j in range(1,i):#对每一个i，从0-i开始找波形
               if nums[i]>nums[j] and dp[j]<0:#与前一个构成up，那么前一个需要是down即负数
                    dp[i]=-dp[j]+1
               elif nums[i]<nums[j] and dp[j]>0:#与前一个构成down，那么前一个需要是up即正数
                    dp[i]=-dp[j]-1
     print(dp)
     res=0
     for m in dp:
          res=max(abs(m),res)
     return res
#print(wiggleMaxLength())




