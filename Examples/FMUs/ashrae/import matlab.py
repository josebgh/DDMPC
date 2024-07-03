import matlab.engine

# Iniciar el motor de MATLAB
eng = matlab.engine.start_matlab()

# Definir las variables en Python
x = 10
y = 20

# # Enviar las variables a MATLAB
eng.workspace['x'] = x
eng.workspace['y'] = y

# Llamar a una función de MATLAB (por ejemplo, 'miFuncion') y obtener el resultado
resultado = eng.miFuncionMatlab(x,y,nargout=1)
print(resultado)

eng.run('miCodigoMatlab.m', nargout=0)
eng.run('miCodigoMatlab2.m', nargout=0)
hola = eng.workspace['hola']
print(hola)

# Cerrar el motor de MATLAB
eng.quit()