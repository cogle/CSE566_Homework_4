
<div>
<h2>Problem 1</h2>
</div>

<div>
    <h3>Hypothesis</h3>
    <p>
    My hypothesis is that the CUDA enabled GPU will perform much, much faster. From
having done the Final Project before this the GPU exhibits extreme performance 
gains over the CPU. While threading is and can be a powerful tool, this is a 
numerical problem the exactly the type of problem the GPU is catered to solving. 
    </p>
</div>


```python
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF 

# Create random data with numpy
import numpy as np
x0 = ['1.50','1.75','2.00','2.25',
      '2.50','2.75','3.00','3.25',
      '3.50','3.75','4.00','4.25',
      '4.50','4.75','5.00']

#15000
y0_cuda = [0.087,0.087,0.087,0.087,0.087]
#17500
y1_cuda = [0.118,0.118,0.109,0.107,0.105]
#20000
y2_cuda = [0.153,0.153,0.137,0.151,0.143]
#22500
y3_cuda = [0.192,0.192,0.173,0.173,0.173]
#25000
y4_cuda = [0.228,0.229,0.214,0.213,0.210]
#27500
y5_cuda = [0.262,0.266,0.249,0.247,0.248]
#30000
y6_cuda = [0.315,0.322,0.300,0.303,0.299]
#32500
y7_cuda = [0.351,0.349,0.374,0.357,0.353]
#35000
y8_cuda = [0.423,0.424,0.401,0.402,0.403]
#37500
y9_cuda = [0.476,0.485,0.468,0.463,0.465]
#40000
y10_cuda = [0.524,0.526,0.522,0.525,0.522]
#42500
y11_cuda = [0.602,0.594,0.585,0.590,0.588]
#45000
y12_cuda = [0.674,0.683,0.665,0.662,0.663]
#47500
y13_cuda = [0.742,0.732,0.732,0.747,0.735]
#50000
y14_cuda = [ 0.807,0.809,0.809,0.808,0.813]


#15000
y0_openmp = [6.322,6.320,6.325,6.321,6.277]
#17500
y1_openmp = [8.616,8.617,8.610,8.605,8.605]
#20000
y2_openmp = [11.250,11.246,11.231,11.241,11.277]
#22500
y3_openmp = [14.229,14.149,14.250,14.205,14.236]
#25000
y4_openmp = [17.496,17.567,17.554,17.548,17.577]
#27500
y5_openmp = [21.252,21.214,21.221,21.272,21.253]
#30000
y6_openmp = [24.833,24.673,24.549,24.509,26.625]
#32500
y7_openmp = [26.702,26.680,26.647,27.306,26.656]
#35000
y8_openmp = [31.461,30.935,31.103,30.945,30.974]
#37500
y9_openmp = [35.766,35.561,35.796,35.493,35.542]
#40000
y10_openmp = [40.699,40.557,40.698,40.443,40.353]
#42500
y11_openmp = [46.025,45.964,45.791,45.784,45.906]
#45000
y12_openmp = [51.348,51.092,51.553,51.477,51.520]
#47500
y13_openmp = [58.077,57.316,57.331,57.495,57.162]
#50000
y14_openmp = [63.223,64.461,63.221,63.564,63.273]

y_grap1 = [np.average(y0_cuda),
           np.average(y1_cuda),
           np.average(y2_cuda),
           np.average(y3_cuda),
           np.average(y4_cuda),
           np.average(y5_cuda),
           np.average(y6_cuda),
           np.average(y7_cuda),
           np.average(y8_cuda),
           np.average(y9_cuda),
           np.average(y10_cuda),
           np.average(y11_cuda),
           np.average(y12_cuda),
           np.average(y13_cuda),
           np.average(y14_cuda)
          ]

y_grap2 = [np.average(y0_openmp),
           np.average(y1_openmp),
           np.average(y2_openmp),
           np.average(y3_openmp),
           np.average(y4_openmp),
           np.average(y5_openmp),
           np.average(y6_openmp),
           np.average(y7_openmp),
           np.average(y8_openmp),
           np.average(y9_openmp),
           np.average(y10_openmp),
           np.average(y11_openmp),
           np.average(y12_openmp),
           np.average(y13_openmp),
           np.average(y14_openmp)
          ]



# Create traces
trace0 = go.Scatter(
    x = x0,
    y = y_grap1,
    mode = 'lines+markers',
    name = 'Cuda',
    error_y=dict(
        type='data',
        array=[
              np.std(y0_cuda),
              np.std(y1_cuda),
              np.std(y2_cuda),
              np.std(y3_cuda),
              np.std(y4_cuda),
              np.std(y5_cuda),
              np.std(y6_cuda),
              np.std(y7_cuda),
              np.std(y8_cuda),
              np.std(y9_cuda),
              np.std(y10_cuda),
              np.std(y11_cuda),
              np.std(y12_cuda),
              np.std(y13_cuda),
              np.std(y14_cuda),
              ],
        visible=True
    )
)


trace1 = go.Scatter(
    x = x0,
    y = y_grap2,
    mode = 'lines+markers',
    name = 'OpenMP',
    error_y=dict(
        type='data',
        array=[
              np.std(y0_openmp),
              np.std(y1_openmp),
              np.std(y2_openmp),
              np.std(y3_openmp),
              np.std(y4_openmp),
              np.std(y5_openmp),
              np.std(y6_openmp),
              np.std(y7_openmp),
              np.std(y8_openmp),
              np.std(y9_openmp),
              np.std(y10_openmp),
              np.std(y11_openmp),
              np.std(y12_openmp),
              np.std(y13_openmp),
              np.std(y14_openmp),
              ],
        visible=True
    )
)


data = [trace0, trace1]
layout=go.Layout(height=1000,
                 title="Runtimes vs. OpenMP or Cuda Submission", 
                 xaxis={'title':'Number of bodies (1x10^4)'}, 
                 yaxis={'title':'Time in Seconds'})

figure=go.Figure(data=data,layout=layout)
py.iplot(figure, filename='Cuda-vs-OpenMP-Runtime')
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~cogle/46.embed" height="1000px" width="100%"></iframe>



<div>
    <a href="https://plot.ly/~cogle/46/" target="_blank" title="Runtimes vs. OpenMP or Cuda Submission" style="display: block; text-align: center;"><img src="https://plot.ly/~cogle/46.png" alt="Runtimes vs. OpenMP or Cuda Submission" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="cogle:46"  src="https://plot.ly/embed.js" async></script>
</div>


<div>
    <h3>Data Analysis</h3>
    <p>
    The results above should come as no surprise. It was expected that the CUDA
    enabled graphics card would fare much better then OpenMP. The graphic card's
    knockout punch comes from its ability to divide the task up amongst many more
    threads. The manner in which the graphic card an OpenMP parallelize a task is
    very different.
    </p>
    <p>
    In OpenMP each thread is assigned a portion of the problem to
    solves; each of these threads then go off and independently solve the problem.
    That means that between any two threads if one was to look at the progress it is
    possible that one might be in a different place than the other. In order to
    support this the CPU has a very large Cache to support the various needs of
    each thread.
    </p>
    <p>
    CUDA's approach is single instruction multiple threads(SIMT). This means that a
    single execution instruction is being broadcast to various units. In CUDA the
    cores of a GPU are collected into groups called streaming multiprocessors.
    These streaming multiprocessors take a groups blocks which have been places into
    an abstraction called a warp and execute the same instruction on a given clock
    cycle.
    </p>
    <p>
    From the two parallel processing paradigms described above we can see that each
    one has its strengths and weaknesses. In the case of having a program that has a
    heavy emphasis on logic, where each thread may not be executing the same
    instruction due to various conditional statements, the first threading paradigm
    might be a better fit. However, if you have lots of data that is going to be
    undergoing the same operation as any other piece of data the SIMT approach is
    much more efficient approach.
    </p>
    <p>
    From this we can begin to understand why the CUDA code would perform so much
    better. This code is simple; there are no complex conditionals it is a simple
    array access and then some calculations. This makes it a prime candidate for
    parallelization across multiple cores. The runtime difference between the two 
    sets of code is fairly extreme, but we can understand why. The biggest 
    bottle-neck for the CUDA program would be the transfer of data to and from the
    CPU otherwise due to the fact that this code lends itself well to the SIMT 
    parallelization ideology.
    </p>
</div>

<div>
<h2>Problem 2</h2>
</div>

<h3>Hypothesis</h3>
<p>
In the second problem we were tasked with looking at how the number of blocks in
the grid would influence the run time of the algorithm. I don't really think it 
will factor into the runtime of the program. So if one was to decrease the 
number of blocks that would certainly decrease the runtime. That would also 
decrease the amount of the problem that is getting solved. So decreasing the 
number of blocks in the grid is out of the question.
</p>


```python
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF 

import numpy as np

x0 = ['118','120','124','128','132','136',
      '140','144','148','152']


#118
y0 = [0.301,0.301,0.304,0.303,0.303]
#120
y1 = [0.297,0.299,0.323,0.297,0.298]
#124
y2 = [0.321,0.295,0.308,0.299,0.299]
#128
y3 = [0.312,0.321,0.299,0.296,0.301]
#132
y4 = [0.317,0.298,0.326,0.296,0.299]
#136
y5 = [0.298,0.300,0.300,0.301,0.298]
#140
y6 = [0.299,0.298,0.300,0.299,.300]
#144
y7 = [0.299,0.298,.302,0.298,0.301]
#148
y8 = [0.299,0.297,0.300,0.297,0.298]
#152
y9 = [0.301,0.304,0.304,0.301,0.314]
#156
y10 = [0.321,0.303,0.300,0.304,0.299]
#160
y11 = [0.317,0.305,0.305,0.305,0.302]
#164
y12 = [0.318,0.301,0.304,0.300,0.304]
#168
y13 = [0.319,0.302,0.301,0.303,0.304]
#172
y14 = [0.323,0.302,0.300,0.301,0.299]
#176
y15 = [0.317,0.305,0.301,0.302,0.302]
#180
y16 = [0.311,0.300,0.323,0.307,0.302]

y_graph = [np.average(y0),
           np.average(y1),
           np.average(y2),
           np.average(y3),
           np.average(y4),
           np.average(y5),
           np.average(y6),
           np.average(y7),
           np.average(y8),
           np.average(y9),
           np.average(y10),
           np.average(y11),
           np.average(y12),
           np.average(y13),
           np.average(y14),
           np.average(y15),
           np.average(y16)
          ]


# Create traces
trace0 = go.Scatter(
    x = x0,
    y = y_graph,
    mode = 'lines+markers',
    name = 'r = vM (Default)',
    error_y=dict(
        type='data',
        array=[
              np.std(y0),
              np.std(y1),
              np.std(y2),
              np.std(y3),
              np.std(y4),
              np.std(y5),
              np.std(y6),
              np.std(y7),
              np.std(y9),
              np.std(y10),
              np.average(y11),
              np.average(y12),
              np.average(y13),
              np.average(y14),
              np.average(y15),
              np.average(y16)
              ],
        visible=True
    )
)

data = [trace0]
layout=go.Layout(height=1000,
                 title="Blocks in Grid vs Runtime", 
                 xaxis={'title':'Number of blocks in the grid'}, 
                 yaxis={'title':'Time(sec)'})

figure=go.Figure(data=data,layout=layout)
py.iplot(figure, filename='Blocks-in-Grid')
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~cogle/50.embed" height="1000px" width="100%"></iframe>



<div>
    <a href="https://plot.ly/~cogle/50/" target="_blank" title="Blocks in Grid vs Runtime" style="display: block; text-align: center;"><img src="https://plot.ly/~cogle/50.png" alt="Blocks in Grid vs Runtime" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="cogle:50"  src="https://plot.ly/embed.js" async></script>
</div>


<div>
    <h3>Data Analysis</h3>
    <p>
    From the results above it is hard to draw a definitive conclusion. I don't think
    that using many blocks will really influence the execution of the program unless
    you use an extreme amount, causing the GPU to have to switch out a lot of
    blocks that would ultimately do nothing. So from the results above we see that
    its really hard to determine what effect modifying just the block size would
    have. Yes we can lower the block size, but then we really wouldn't be solving
    the full problem and while achieving a lower runtime, the algorithm would fail
    correctness.
    </p>
<div>

<div>
<h2>Problem 3</h2>
</div>

<h3>Hypothesis</h3>
<p>
Based upon my research the internet has told me that when choosing the number of
threads, a multiple of 32 performs very well. This is because the maximum number
of threads in a warp is 32; therefore for max utility of a warp we should aim 
for a multiple of 32; this will ensure that each warp is filled to capacity. 
</p>


```python
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF 

import numpy as np

x0 = ['8','16','32','48','64','80','96','112',
      '128','144','160','176','192','208','224',
      '240','256','272','288','320','352','384',
      '416','448','480','512']


#8
y0 = [1.129,1.132,1.129,1.127,1.127]
#16
y1 = [0.580,0.582,0.570,0.570,0.570]
#32
y2 = [0.310,0.318,0.296,0.296,0.295]
#48
y3 = [0.389,0.391,0.391,0.388,0.410]
#64
y4 = [0.309,0.318,0.298,0.299,0.299]
#80
y5 = [0.372,0.375,0.356,0.353,0.353]
#96
y6 = [0.311,0.298,0.314,0.299,0.297]
#112
y7 = [0.314,0.316,0.300,0.297,0.296]
#128
y8 = [0.313,0.299,0.312,0.299,0.298]
#144
y9 = [0.342,0.346,0.333,0.335,0.331]
#160
y10 = [0.319,0.298,0.298,0.299,0.298]
#176
y11 = [0.322,0.322,0.323,0.322,0.321]
#192
y12 = [0.317,0.315,0.299,0.297,0.299]
#208
y13 = [0.343,0.329,0.322,0.321,0.322]
#224
y14 = [0.316,0.321,0.303,0.304,0.304]
#240
y15 = [0.343,0.317,0.319,0.341,0.342]
#256
y16 = [0.313,0.317,0.302,0.299,0.303]
#272
y17 = [0.315,0.313,0.317,0.331,0.337]
#288
y18 = [0.304,0.312,0.302,0.317,0.302]
#320
y19 = [0.320,0.306,0.297,0.298,0.299]
#352
y20 = [0.322,0.303,0.324,0.304,0.304]
#384
y21 = [0.312,0.297,0.328,0.319,0.298]
#416
y22 = [0.327,0.321,0.305,0.304,0.303]
#448
y23 = [0.328,0.326,0.303,0.307,0.303]
#
y24 = [0.298,0.298,0.299,0.319,0.298]
#
y25 = [0.325,0.305,0.300,0.307,0.307]

y_graph = [np.average(y0),
           np.average(y1),
           np.average(y2),
           np.average(y3),
           np.average(y4),
           np.average(y5),
           np.average(y6),
           np.average(y7),
           np.average(y8),
           np.average(y9),
           np.average(y10),
           np.average(y11),
           np.average(y12),
           np.average(y13),
           np.average(y14),
           np.average(y15),
           np.average(y16),
           np.average(y17),
           np.average(y18),
           np.average(y19),
           np.average(y20),
           np.average(y21),
           np.average(y22),
           np.average(y23),
           np.average(y24),
           np.average(y25)
          ]


# Create traces
trace0 = go.Scatter(
    x = x0,
    y = y_graph,
    mode = 'lines+markers',
    name = 'r = vM (Default)',
    error_y=dict(
        type='data',
        array=[
              np.std(y0),
              np.std(y1),
              np.std(y2),
              np.std(y3),
              np.std(y4),
              np.std(y5),
              np.std(y6),
              np.std(y7),
              np.std(y8),
              np.std(y9),
              np.std(y10),
              np.std(y11),
              np.std(y12),
              np.std(y13),
              np.std(y14),
              np.std(y15),
              np.std(y16),
              np.std(y17),
              np.std(y18),
              np.std(y19),
              np.std(y20),
              np.std(y21),
              np.std(y22),
              np.std(y23),
              np.std(y24),
              np.std(y25)
              ],
        visible=True
    )
)

data = [trace0]
layout=go.Layout(height=1000,
                 title="Threads in block vs Runtime", 
                 xaxis={'title':'Threads in Block'}, 
                 yaxis={'title':'Time(sec)'})

figure=go.Figure(data=data,layout=layout)
py.iplot(figure, filename='Threads-Block')
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~cogle/52.embed" height="1000px" width="100%"></iframe>



<div>
    <a href="https://plot.ly/~cogle/52/" target="_blank" title="Threads in block vs Runtime" style="display: block; text-align: center;"><img src="https://plot.ly/~cogle/52.png" alt="Threads in block vs Runtime" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="cogle:52"  src="https://plot.ly/embed.js" async></script>
</div>


<div>
<h3>Data Analysis</h3>
<p>
So the results for this particular test are pretty neat. They do indeed prove
that using a multiple of 32 is the best choice for the particular problem;
however, as the number of threads per block increases this advantage decreases.
</p>
<p>
From the results above and comparing them against the previous problem we see
that choosing the correct number of threads is a far more critical component
when using CUDA to run code. Threads determine how the code is divvied up to the
GPU. Poor choices in work allocation translate to poor runtime results as the
above graph demonstrates. Having more than one warp with unused cores results
in a lower GPU utilization overall, which translates into wasted instructions.
</p>
<p>
Creating parallel code is not just about being able to break apart code and run
it in parallel. Rather, as this problem demonstrates parallelization is about
being able to write parallel code that optimally uses the system's hardware. 
This problem highlight the importance that choice and the difference poor 
hardware utilization makes in running parallelizable code. 
</p>
</div>

<div>
<h2>Problem 4</h2>
</div>


```python
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF 

# Create random data with numpy
import numpy as np
x0 = ['1.50','1.75','2.00','2.25',
      '2.50','2.75','3.00','3.25',
      '3.50','3.75','4.00','4.25',
      '4.50','4.75','5.00']

#15000
y0_cuda = [0.087,0.087,0.087,0.087,0.087]
#17500
y1_cuda = [0.118,0.118,0.109,0.107,0.105]
#20000
y2_cuda = [0.153,0.153,0.137,0.151,0.143]
#22500
y3_cuda = [0.192,0.192,0.173,0.173,0.173]
#25000
y4_cuda = [0.228,0.229,0.214,0.213,0.210]
#27500
y5_cuda = [0.262,0.266,0.249,0.247,0.248]
#30000
y6_cuda = [0.315,0.322,0.300,0.303,0.299]
#32500
y7_cuda = [0.351,0.349,0.374,0.357,0.353]
#35000
y8_cuda = [0.423,0.424,0.401,0.402,0.403]
#37500
y9_cuda = [0.476,0.485,0.468,0.463,0.465]
#40000
y10_cuda = [0.524,0.526,0.522,0.525,0.522]
#42500
y11_cuda = [0.602,0.594,0.585,0.590,0.588]
#45000
y12_cuda = [0.674,0.683,0.665,0.662,0.663]
#47500
y13_cuda = [0.742,0.732,0.732,0.747,0.735]
#50000
y14_cuda = [ 0.807,0.809,0.809,0.808,0.813]


y_grap1 = [np.average(y0_cuda),
           np.average(y1_cuda),
           np.average(y2_cuda),
           np.average(y3_cuda),
           np.average(y4_cuda),
           np.average(y5_cuda),
           np.average(y6_cuda),
           np.average(y7_cuda),
           np.average(y8_cuda),
           np.average(y9_cuda),
           np.average(y10_cuda),
           np.average(y11_cuda),
           np.average(y12_cuda),
           np.average(y13_cuda),
           np.average(y14_cuda)
          ]




# Create traces
trace0 = go.Scatter(
    x = x0,
    y = y_grap1,
    mode = 'lines+markers',
    name = 'Cuda',
    error_y=dict(
        type='data',
        array=[
              np.std(y0_cuda),
              np.std(y1_cuda),
              np.std(y2_cuda),
              np.std(y3_cuda),
              np.std(y4_cuda),
              np.std(y5_cuda),
              np.std(y6_cuda),
              np.std(y7_cuda),
              np.std(y8_cuda),
              np.std(y9_cuda),
              np.std(y10_cuda),
              np.std(y11_cuda),
              np.std(y12_cuda),
              np.std(y13_cuda),
              np.std(y14_cuda),
              ],
        visible=True
    )
)


data = [trace0]
layout=go.Layout(height=1000,
                 title="Runtimes vs. Bodies", 
                 xaxis={'title':'Number of bodies (1x10^4)'}, 
                 yaxis={'title':'Time in Seconds'})

figure=go.Figure(data=data,layout=layout)
py.iplot(figure, filename='Cuda-Runtime')
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~cogle/56.embed" height="1000px" width="100%"></iframe>



<div>
    <a href="https://plot.ly/~cogle/56/" target="_blank" title="Runtimes vs. Cuda" style="display: block; text-align: center;"><img src="https://plot.ly/~cogle/56.png" alt="Runtimes vs. Cuda" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="cogle:56"  src="https://plot.ly/embed.js" async></script>
</div>

<div>
<h3>Data Analysis</h3>
<p>
This is the same graph from the very first problem, minus the CPU code. In this
graph we see a very slow increase in time that it takes for the GPU to complete
the problem. Simply put the GPU is able to handle this problem very well. As
outlined in the first problem's analysis, this problem is a very good candidate
for parallelization. The graph is interesting because it appears that it is a
quadratic increase in runtime, despite a linear increase in the problem size.
I can not definitively say what causes to resemble a quadratic function, but my
guess is that it comes from having to send the problem from the CPU to the GPU
and visa-versa. What this graph does highlight is the blistering speed at which
the GPU can solve a problem such as this.
</p>
</div>

<div>
<h2>Problem 5</h2>
</div>


```python
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF 

# Create random data with numpy
import numpy as np
x0 = ['1.50','1.75','2.00','2.25',
      '2.50','2.75','3.00','3.25',
      '3.50','3.75','4.00','4.25',
      '4.50','4.75','5.00']

#15000
y0_cuda = [0.087,0.087,0.087,0.087,0.087]
#17500
y1_cuda = [0.118,0.118,0.109,0.107,0.105]
#20000
y2_cuda = [0.153,0.153,0.137,0.151,0.143]
#22500
y3_cuda = [0.192,0.192,0.173,0.173,0.173]
#25000
y4_cuda = [0.228,0.229,0.214,0.213,0.210]
#27500
y5_cuda = [0.262,0.266,0.249,0.247,0.248]
#30000
y6_cuda = [0.315,0.322,0.300,0.303,0.299]
#32500
y7_cuda = [0.351,0.349,0.374,0.357,0.353]
#35000
y8_cuda = [0.423,0.424,0.401,0.402,0.403]
#37500
y9_cuda = [0.476,0.485,0.468,0.463,0.465]
#40000
y10_cuda = [0.524,0.526,0.522,0.525,0.522]
#42500
y11_cuda = [0.602,0.594,0.585,0.590,0.588]
#45000
y12_cuda = [0.674,0.683,0.665,0.662,0.663]
#47500
y13_cuda = [0.742,0.732,0.732,0.747,0.735]
#50000
y14_cuda = [0.807,0.809,0.809,0.808,0.813]


#15000
y0_opt = [0.086,0.086,0.086,0.087,0.087]
#17500
y1_opt = [0.115,0.105,0.105,0.107,0.106]
#20000
y2_opt = [0.147,0.136,0.135,0.135,0.135]
#22500
y3_opt = [0.187,0.172,0.172,0.170,0.172]
#25000
y4_opt = [0.232,0.210,0.211,0.211,0.210]
#27500
y5_opt = [0.259,0.247,0.246,0.247,0.246]
#30000
y6_opt = [0.309,0.302,0.300,0.301,0.296]
#32500
y7_opt = [.361,0.349,0.368,0.350,0.350]
#35000
y8_opt = [0.416,0.402,0.402,0.402,0.402]
#37500
y9_opt = [0.467,0.465,0.462,0.462,0.457]
#40000
y10_opt = [0.537,0.524,0.527,0.526,0.529]
#42500
y11_opt = [0.602,0.593,0.592,0.592,0.588]
#45000
y12_opt = [0.667,0.661,0.657,0.657,0.656]
#47500
y13_opt = [0.739,0.735,0.736,0.737,0.734]
#50000
y14_opt = [0.810,0.810,0.811,0.812,0.814]

y_grap1 = [np.average(y0_cuda),
           np.average(y1_cuda),
           np.average(y2_cuda),
           np.average(y3_cuda),
           np.average(y4_cuda),
           np.average(y5_cuda),
           np.average(y6_cuda),
           np.average(y7_cuda),
           np.average(y8_cuda),
           np.average(y9_cuda),
           np.average(y10_cuda),
           np.average(y11_cuda),
           np.average(y12_cuda),
           np.average(y13_cuda),
           np.average(y14_cuda)
          ]

y_grap2 = [np.average(y0_opt),
           np.average(y1_opt),
           np.average(y2_opt),
           np.average(y3_opt),
           np.average(y4_opt),
           np.average(y5_opt),
           np.average(y6_opt),
           np.average(y7_opt),
           np.average(y8_opt),
           np.average(y9_opt),
           np.average(y10_opt),
           np.average(y11_opt),
           np.average(y12_opt),
           np.average(y13_opt),
           np.average(y14_opt)
          ]



# Create traces
trace0 = go.Scatter(
    x = x0,
    y = y_grap1,
    mode = 'lines+markers',
    name = 'Cuda',
    error_y=dict(
        type='data',
        array=[
              np.std(y0_cuda),
              np.std(y1_cuda),
              np.std(y2_cuda),
              np.std(y3_cuda),
              np.std(y4_cuda),
              np.std(y5_cuda),
              np.std(y6_cuda),
              np.std(y7_cuda),
              np.std(y8_cuda),
              np.std(y9_cuda),
              np.std(y10_cuda),
              np.std(y11_cuda),
              np.std(y12_cuda),
              np.std(y13_cuda),
              np.std(y14_cuda),
              ],
        visible=True
    )
)


trace1 = go.Scatter(
    x = x0,
    y = y_grap2,
    mode = 'lines+markers',
    name = 'Optimized CUDA',
    error_y=dict(
        type='data',
        array=[
              np.std(y0_opt),
              np.std(y1_opt),
              np.std(y2_opt),
              np.std(y3_opt),
              np.std(y4_opt),
              np.std(y5_opt),
              np.std(y6_opt),
              np.std(y7_opt),
              np.std(y8_opt),
              np.std(y9_opt),
              np.std(y10_opt),
              np.std(y11_opt),
              np.std(y12_opt),
              np.std(y13_opt),
              np.std(y14_opt),
              ],
        visible=True
    )
)


data = [trace0, trace1]
layout=go.Layout(height=1000,
                 title="Runtimes vs. OpenMP or Cuda Submission", 
                 xaxis={'title':'Number of bodies (1x10^4)'}, 
                 yaxis={'title':'Time in Seconds'})

figure=go.Figure(data=data,layout=layout)
py.iplot(figure, filename='Cuda-vs-Opt-Runtime')

```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~cogle/54.embed" height="1000px" width="100%"></iframe>



<div>
    <a href="https://plot.ly/~cogle/54/" target="_blank" title="Runtimes vs. OpenMP or Cuda Submission" style="display: block; text-align: center;"><img src="https://plot.ly/~cogle/54.png" alt="Runtimes vs. OpenMP or Cuda Submission" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="cogle:54"  src="https://plot.ly/embed.js" async></script>
</div>


<div>
<img src="CSE566_Homework_4/Results/images/Opt.PNG">
<p>
Above is the optimized code in order to optimize the code I moved the for loop
that ran was being ran outside from main into the CUDA code block. In order to
maintain correctness I had to put __syncthreads() code block in to ensure that
each thread was able to read all the values before they were updated. After all
threads had finished the updates to their position was recorded.
</p>
<p>
As we can see from the chart above we were able to achieve minimal performance
gains. While they aren't as drastic as I hoped, partially due to the
__syncthreads() call, it does show some performance gains. The reason that this
does lead to performance gains is that we have moved the iterative code into the
GPU function call. This allows us to forgo having to call the loop, simply once
all threads have been updated we update our value. This allows us to gain some
performance as we no longer have to create a for loop and run through it
one-by-one. Instead the warp can take care of it.
</p>
<div>


```python

```
