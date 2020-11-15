from main import Net

def heading(msg):
    print(('='*20) + ' ' + msg + ' ' + ('='*20))


# Simple handcrafted test of net.set_from_params
heading('Test handcrafted')
net = Net.random((3,2,4,6))
p = net.params()
p[-1] = 100
p[0]  = 200

print([w.shape for w in net.weights])
net = Net.from_params(p, net.layers)
print(f'Biases {[b.shape for b in net.biases]}\n', net.biases)
print(f'Weights {[w.shape for w in net.weights]}\n', net.weights)
assert net.biases[0][0] == 200
got = net.weights[-1][-1][-1]
assert got == 100, f'want 100 got {got}'


# ===========================
# global test, add 20 to all (does not test structure!)

heading('Test add 20')
net = Net.random((3,2,4,5))
p = net.params()
p = p + 20

net = Net.from_params(p, net.layers)
print('Biases\n', net.biases)
print('Weights\n', net.weights)
for b in net.biases:
    assert (b > 10).all()

for w in net.weights:
    assert (w > 10).all()
