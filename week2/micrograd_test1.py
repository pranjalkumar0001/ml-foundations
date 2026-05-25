import micrograd_engine as mg
value = mg.value
x  = value(2.0)
w1 = value(0.5)
w2 = value(0.8)
t  = value(1.0)

h    = x * w1
a    = h.sigmoid()
y    = a * w2
diff = y + value(-1.0) * t
loss = diff * diff

loss.backward()

print(x)
print(w1)
print(w2)