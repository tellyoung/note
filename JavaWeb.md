# Servlet









# Spring











# SpringBoot

- 自动生成以下文件：
  1. 程序的主启动类
  2. 一个 application.properties 配置文件
  3. 一个 测试类
  4. 一个 pom.xml



## 注解

- @Component

  注册bean到容器中

- @Value

  给bean注入属性值

  ```java
  
  ```

  

- @Autowired

  自动注入

  ```java
  
  ```

  



- @ConfigurationProperties

  默认从全局配置文件中获取值

  将配置文件中配置的每一个属性的值，映射到这个组件中；

  ```java
  
  ```

  

- @PropertySource

  加载指定的配置文件

  ```java
  // 指定加载person.properties文件
  @PropertySource(value = "classpath:person.properties")
  @Component //注册bean
  public class Person {
      @Value("${name}")
      private String name;
  }
  ```

  



## 配置文件

- application.properties (默认配置文件)
  - 语法结构 ： key=value

1. 新建一个实体类User

```java
@Component //注册bean
public class User {
    private String name;
    private int age;
    private String sex;
}
```

2. 编辑配置文件 user.properties

```java
user1.name=kuangshen
user1.age=18
user1.sex=男
```

3. 在User类上使用@Value来进行注入！

```java
@Component //注册bean
@PropertySource(value = "classpath:user.properties")
public class User {
    //直接使用@value
    @Value("${user.name}") //从配置文件中取值
    private String name;
    @Value("#{9*2}") // #{SPEL} Spring表达式
    private int age;
    @Value("男") // 字面量
    private String sex;
}
```

4. 测试

```java
@SpringBootTest
class DemoApplicationTests {
    @Autowired
    User user;
    
    @Test
    public void contextLoads() {
    	System.out.println(user);
    }
}
```





```java

```



- application.yml

  - 语法结构 ：key：空格 value

  

- pom.xml

```xml
<!-- 父依赖 -->
<parent>
<groupId>org.springframework.boot</groupId>
<artifactId>spring-boot-starter-parent</artifactId>
<version>2.2.5.RELEASE</version>
<relativePath/>
</parent>
<dependencies>
<!-- web场景启动器 -->
<dependency>
<groupId>org.springframework.boot</groupId>
<artifactId>spring-boot-starter-web</artifactId>
</dependency>
<!-- springboot单元测试 -->
<dependency>
<groupId>org.springframework.boot</groupId>
<artifactId>spring-boot-starter-test</artifactId>
<scope>test</scope>
<!-- 剔除依赖 -->
<exclusions>
<exclusion>
<groupId>org.junit.vintage</groupId>
<artifactId>junit-vintage-engine</artifactId>
</exclusion>
</exclusions>
</dependency>
</dependencies>
<build>
<plugins>
<!-- 打包插件 -->
<plugin>
<groupId>org.springframework.boot</groupId>
<artifactId>spring-boot-maven-plugin</artifactId>
</plugin>
</plugins>
</build>
```



## 注入

### 注解注入

```java
@Component //注册bean
public class Dog {
    @Value("阿黄")
    private String name;
    @Value("18")
    private Integer age;
}
```





### yaml注入

```java
/*
    @ConfigurationProperties作用：
    将配置文件中配置的每一个属性的值，映射到这个组件中；
    告诉SpringBoot将本类中的所有属性和配置文件中相关的配置进行绑定
    参数prefix=“person”:将配置文件中的person下面的所有属性一一对应
*/
@Component //注册bean到容器中
@ConfigurationProperties(prefix="person")
public class Person {
    private String name;
    private Integer age;
    private Boolean happy;
    private Date birth;
    private Map<String,Object> maps;
    private List<Object> lists;
    private Dog dog;
}
```



- yaml文件

  ```yaml
  person:
      name: qinjiang
      age: 3
      happy: false
      birth: 2000/01/01
      maps: {k1: v1,k2: v2}
      lists:
          - code
          - girl
          - music
      dog:
          name: 旺财
          age: 1
  ```

  

## JSR303数据校验



```java
@Component //注册bean
@ConfigurationProperties(prefix = "person")
@Validated //数据校验
public class Person {
    @NotNull(message="名字不能为空")
    private String userName;
    
    @Max(value=120,message="年龄最大不能查过120")
    private int age;
    
    @Email(message="邮箱格式错误") //name必须是邮箱格式
    private String email;
}
```



```java
# 空检查
@Null 验证对象是否为null
@NotNull 验证对象是否不为null, 无法查检长度为0的字符串
@NotBlank 检查约束字符串是不是Null还有被Trim的长度是否大于0,只对字符串,且会去掉前后空格.
@NotEmpty 检查约束元素是否为NULL或者是EMPTY.
    
# Booelan检查
@AssertTrue 验证 Boolean 对象是否为 true
@AssertFalse 验证 Boolean 对象是否为 false
    
# 长度检查
@Size(min=, max=) 验证对象（Array,Collection,Map,String）长度是否在给定的范围之内
@Length(min=, max=) string is between min and max included.
    
# 日期检查
@Past 验证 Date 和 Calendar 对象是否在当前时间之前
@Future 验证 Date 和 Calendar 对象是否在当前时间之后
@Pattern 验证 String 对象是否符合正则表达式的规则
```









## Web

### Web注解

@RestController



@RequestMapping()





### 编写HTTP接口

1. 在主程序的同级目录下，新建一个controller包，一定要在同级目录下，否则识别不到
2. 在包中新建一个HelloController类

```java
@RestController
public class HelloController {
    @RequestMapping("/hello")
    public String hello() {
        return "Hello World";
    }
}
```







## 打包

- 将项目打成jar包，点击 maven的 package

```xml
<!--
在工作中,很多情况下打包是不想执行测试用例的,跳过测试用例
-->
<plugin>
    <groupId>org.apache.maven.plugins</groupId>
    <artifactId>maven-surefire-plugin</artifactId>
    <configuration>
        <!--跳过项目运行测试用例-->
        <skipTests>true</skipTests>
    </configuration>
</plugin>
```



- 如果打包成功，则会在target目录下生成一个 jar 包

- `cmd：java -jar [包名].jar`



## 启动图案

如何更改启动时显示的字符拼成的字母:

- resources 目录下新建一个banner.txt 即可。
  图案可以到：https://www.bootschool.net/ascii 这个网站生成，然后拷贝到文件中即可！





# yaml语法

1、空格不能省略
2、以缩进来控制层级关系，只要是左边对齐的一列数据都是同一个层级的。
3、属性和值的大小写都是十分敏感的。

```yaml
server：
	prot: 8080
```



## 字面量

> 普通的值 [ 数字，布尔值，字符串 ]

```yaml
k: v
```

- “ ”  双引号，不会转义字符串里面的特殊字符， 特殊字符会作为本身想表示的意思；
  比如 ： `name: "kuang \n shen"` 输出 ： `kuang 换行 shen`
- '' 单引号，会转义特殊字符 ， 特殊字符最终会变成和普通字符一样输出
  比如 ： `name: ‘kuang \n shen’`  输出 ： `kuang \n shen`



## 对象、Map（键值对）

```yaml
# 对象、Map格式
k:
	v1:
	v2:
	
student:
	name: qinjiang
	age: 3

# 行内写法
student: {name: qinjiang,age: 3}
```



## 数组（ List、set ）

```yaml
pets:
	- cat
	- dog
	- pig

# 行内写法
pets: [cat,dog,pig]
```



## 占位符

```yaml
person:
    name: qinjiang${random.uuid} # 随机uuid
    age: ${random.int} # 随机int
    happy: false
    birth: 2000/01/01
    maps: {k1: v1,k2: v2}
    lists:
        - code
        - girl
        - music
    dog:
    # 引用person.hello 的值，如果不存在就用 ：后面的值，即 other，然后拼接上_旺财
    name: ${person.hello:other}_旺财
    age: 1
```





# MySql





































































































