import pickle
print("Este es mi programa de prueba :)")
for i in range(10):
	print("Prueba ", i, ":)"*(i+1))
print("Fin")

with open("resultadoPrueba.txt", 'wb') as f:
    pickle.dump('Holi', f)