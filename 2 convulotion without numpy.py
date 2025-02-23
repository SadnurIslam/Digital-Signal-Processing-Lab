x = [1,2,3,4]
h = [1,1,0.5]
y = []

for i in range(len(x)+len(h)-1):
    y.append(0)
    
for i in range (len(h)):
    for j in range (len(x)):
        y[i+j] += h[i]*x[j]
        
print("Input Sequence x(n):", x)
print("Impulse Response h(n):", h)
print("Convolution Result y(n):", y)
