JAVAC=javac
JAVA=java
SRC=$(wildcard *.java)
CLASSES=$(SRC:.java=.class)

all: compile

compile:
	$(JAVAC) $(SRC)

clean:
	rm -f *.class *.jar

jar: compile
	jar cfe GPArithmetic.jar GPArithmetic *.class
	jar cfe GPDecisionTree.jar GPDecisionTree *.class
	jar cfe MultiRunEvaluator.jar MultiRunEvaluator *.class

run-task1:
	$(JAVA) Task1Test

run-task2:
	$(JAVA) Task2Test