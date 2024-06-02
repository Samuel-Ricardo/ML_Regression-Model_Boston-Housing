# Linear Regression Model | Boston-Housing


![1_FHQOSHMMT07CbXpklk1Ehw](https://github.com/Samuel-Ricardo/ML_Regression-Model_Boston-Housing/assets/63983021/ba2e1081-b63f-444e-8050-1017d3a28e27)

<br>

<h4 align="center" >🚀 🟥 Linear Regression Model | Boston-Housing 🟥 🚀</h4>

<h4 align="center">
Artificial Intelligence Regression Model for Pricing Housing in Boston
</h4>

#

<p align="center">
  |&nbsp;&nbsp;
  <a style="color: #8a4af3;" href="#project">Overview</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
  <a style="color: #8a4af3;" href="#techs">Technologies</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
  <a style="color: #8a4af3;" href="#app">Project</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;
  <a style="color: #8a4af3;" href="#run-project">Run</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;
  <a style="color: #8a4af3;" href="#author">Author</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
</p>

#

<h1 align="center">
  
  <a href="https://github.com/Samuel-Ricardo">
    <img src="https://img.shields.io/static/v1?label=&message=Samuel%20Ricardo&color=black&style=for-the-badge&logo=GITHUB"/>
  </a>

  <a herf="https://www.instagram.com/samuel_ricardo.ex/">
    <img src='https://img.shields.io/static/v1?label=&message=Samuel.ex&color=black&style=for-the-badge&logo=instagram'/> 
  </a>

  <a herf='https://www.linkedin.com/in/samuel-ricardo/'>
    <img src='https://img.shields.io/static/v1?label=&message=Samuel%20Ricardo&color=black&style=for-the-badge&logo=LinkedIn'/> 
  </a>

</h1>

<br>

<p id="project"/>

<br>

<h2>  | :artificial_satellite: About:  </h2>

<p align="justify">
  I took on the classic challenge of calculating the price of a house in Boston, so I built a Linear Regression model that considers data from the region to evaluate this house. I used Google Colab for this.
</p>

<br>

<h2 id="techs">
  :building_construction: | Technologies and Concepts Studied:
</h2>


> <a href='https://www.tensorflow.org/'> <img width="200px" src="https://github.com/Samuel-Ricardo/ML_Regression-Model_Boston-Housing/assets/63983021/d9ed0c56-4fde-4285-89e5-7ec89bdfab76" /> </a>

- Python
- Google Colab
- Tensorflow
- Keras
- Numpy
- pandas
- tensorflow-addons
- keras-utils
- Matplotlib
- Linear regression
- Graphics
  
> Among Others...

#

<br>

<h2 id="app">
  🧠 | Project:
</h2>

<br/>


To calculate, I took these data from [Keras](https://keras.io/2.15/api/datasets/boston_housing/) as a basis:

<br>

Boston house price data from Harrison, D. and Rubinfeld, D.L. for the article 'Hedonic Prices and the Demand for Clean Air', J. Environ. Economics and Management, vol.5, 81-102, 1978.

<br>

- Per capita crime rate by city.
- Proportion to residential land zoned in lots over 25,000 square feet.
- Proportion in relation to hectares of non-retail land by city
  - What is the relationship between residential land and commercial land
- Proportion in relation to land allocated with rivers
- Concentration of nitric oxides measured on the ground (parts per 10 million)
  - Amount of waste
- Average number of rooms per dwelling in the same subdivision
  - Quite common to have more rooms left to rent to guests
  - These extra rooms are usually planned / estimated during construction
- Proportion of occupied units (subdivisions) and built before 1940
  - Get a sense of how much that area evolved at a specific time
- Weighted distances to five Boston employment centers
  - Distance from subdivision to centers
- Accessibility index to radial highways
- Average value property tax rate (scaled by 10,000)
- Ratio of students per teacher by city
- Proportion of black people by city (per thousand inhabitants)
- Lowest status percentage of the population
- Average home value (scaled by 1,000)

<br>

The goal is to use linear regression to find the best value of owner-occupied homes at 1,000 USD.

#

I created a sequential model with 3 Dense Layers, where the last one is the calculation of the result output, in total we have more than 120 neurons fully connected with more than 5000 parameters, 20 KB in total.

![image](https://github.com/Samuel-Ricardo/ML_Regression-Model_Boston-Housing/assets/63983021/873d7dc3-48c1-479b-b224-3fa762453dfd)

![image](https://github.com/Samuel-Ricardo/ML_Regression-Model_Boston-Housing/assets/63983021/35584d19-c825-49b7-9d6e-f4d7cced309d)

#

I also configured an Optimizer algorithm for better results and corrections in each processing cycle.

![image](https://github.com/Samuel-Ricardo/ML_Regression-Model_Boston-Housing/assets/63983021/44214216-86a5-412f-82d6-c4e7482d57c4)


## ✅ | Results:

I programmed 100 processing epochs (cycles) and the model exceeded more than 90% accuracy with an average absolute error of 1.9%

![image](https://github.com/Samuel-Ricardo/ML_Regression-Model_Boston-Housing/assets/63983021/47a0b92b-1c51-468c-91f7-f392a0eb898e)

<br>

In a graphical view by processing epochs:

![image](https://github.com/Samuel-Ricardo/ML_Regression-Model_Boston-Housing/assets/63983021/98133b77-2b75-409b-bf48-0456222a6e61)

![image](https://github.com/Samuel-Ricardo/ML_Regression-Model_Boston-Housing/assets/63983021/31217b88-3359-4708-8def-9181dd4b10cc)


<h2 id="run-project"> 
   👨‍💻 | How to use
</h2>

<br>

Import the `Boston_Housing_Regressor.ipynb` file in a Python Notebook App like Jupyter or Google Colab and run cell by cell.

- Run only the cell with `!pip install` commands in `[Setup and Install Deps]` section.
- Now, jump to `[Linear Regression | Boston Housing]` Section and run cells in order.


#

<br>

<h2 id="author">
  :octocat: | Author:  
</h2>

> <a target="_blank" href="https://www.linkedin.com/in/samuel-ricardo/"> <img width="350px" src="https://github.com/Samuel-Ricardo/bolao-da-copa/blob/main/readme_files/IMG_20220904_220148_188.jpg?raw=true"/> <br> <p> <b> - Samuel Ricardo</b> </p></a>

<h1>
  <a herf='https://github.com/Samuel-Ricardo'>
    <img src='https://img.shields.io/static/v1?label=&message=Samuel%20Ricardo&color=black&style=for-the-badge&logo=GITHUB'> 
  </a>
  
  <a herf='https://www.instagram.com/samuel_ricardo.ex/'>
    <img src='https://img.shields.io/static/v1?label=&message=Samuel.ex&color=black&style=for-the-badge&logo=instagram'> 
  </a>
  
  <a herf='https://twitter.com/SamuelR84144340'>
    <img src='https://img.shields.io/static/v1?label=&message=Samuel%20Ricardo&color=black&style=for-the-badge&logo=twitter'> 
  </a>
  
   <a herf='https://www.linkedin.com/in/samuel-ricardo/'>
    <img src='https://img.shields.io/static/v1?label=&message=Samuel%20Ricardo&color=black&style=for-the-badge&logo=LinkedIn'> 
  </a>
</h1>
