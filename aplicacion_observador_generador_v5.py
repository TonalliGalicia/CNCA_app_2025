import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import nidaqmx
import re
import ast


"""
Importante: Este código se ha ejecutado en Spyder, instalado a su vez mediante
Anaconda. 
Cómo implementar el código:
1.- Se debe de tener instalado NI-MAX y asegurarse de que el dispositivo DAQ es
reconocido. Si esto no se cumple entonces es necesario verificar los 
controladores de la DAQ. Nota: No es necesario tener instalado LabView, además
de que NI-MAX es gratuito.

2.- Una vez se ha asegurado la comunicación por la DAQ mediante NI-MAX, hay que
prestar atención al nombre/número del dispositivo. Este código contempla que la
DAQ enlazada se reconoció como "Dev1". En NI-MAX se puede identificar y cambiar
este nombre a "Dev1" en caso de requerirlo. IMPORTANTE: considero que es más 
fácil cambiar el nombre en NI-MAX en lugar de hacerlo en el código, aunque 
tambien es posible.

3.- Es posible que se presente el problema de que no se encuentra instalado el 
módulo "nidaqmx", para solucionarlo seguir estos pasos:
    
    3.1 En Spyder, dentro de la terminal ingresar (copiar y pegar):
            import sys
            print(sys.executable)
        aparecerá una ruta que es la que está usando Spyder. Hay que copiar
        esta ruta (se usará en 3.4).
    3.2 Abrir "Anaconda Prompt"
    3.3 Actualizar "pip" y "conda" mediante (copiar y pegar):
        conda update conda
        conda update pip
    3.4 Instalar nidaqmax usando pip de Anaconda con el código (reemplazando la
        ruta con la que se obtuvo en 3.1, se muestra un ejemplo):
        C:/Users/TecNM/anaconda3/python.exe -m pip install nidaqmx
    3.5 Por último verificar la instalación con (copiar y pegar, reemplazando
        la ruta):
        C:/Users/TecNM/anaconda3/python.exe -m pip show nidaqmx
Eso es todo.
"""


class ObserverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Observador de Estados con DAQ")
        
        # Variables de control
        self.running = False
        self.threads = []
        
        # Variables para DAQ
        self.in1 = 0.0
        self.in2 = 0.0
        
        # Crear interfaz
        self.create_widgets()
    
    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # Configuración de simulación
        sim_frame = ttk.LabelFrame(main_frame, text="Configuración de Simulación", padding="10")
        sim_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        ttk.Label(sim_frame, text="Paso (h):").grid(row=0, column=0, sticky="w")
        self.h_entry = ttk.Entry(sim_frame)
        self.h_entry.insert(0, "0.005")
        self.h_entry.grid(row=0, column=1, sticky="w")
        
        ttk.Label(sim_frame, text="Duración (s):").grid(row=1, column=0, sticky="w")
        self.duration_entry = ttk.Entry(sim_frame)
        self.duration_entry.insert(0, "40")
        self.duration_entry.grid(row=1, column=1, sticky="w")
        
        # Condiciones iniciales
        self.ci_frame = ttk.LabelFrame(main_frame, text="Condiciones Iniciales", padding="10")
        self.ci_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        ttk.Label(self.ci_frame, text="Valores (separados por comas):").grid(row=0, column=0, sticky="w")
        self.ci_entry = ttk.Entry(self.ci_frame)
        self.ci_entry.insert(0, "1,5,4")
        self.ci_entry.grid(row=0, column=1, sticky="ew")
        
        # Límites de estados
        self.limits_frame = ttk.LabelFrame(main_frame, text="Matriz de Límites de Estados", padding="10")
        self.limits_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        example_limits = "[[-18.220797, 18.829984],\n[-24.589444, 25.924115],\n[5.083875, 46.496289]]"
        self.limits_text = scrolledtext.ScrolledText(self.limits_frame, width=40, height=5)
        self.limits_text.insert(tk.END, example_limits)
        self.limits_text.grid(row=0, column=0, sticky="nsew")
        
        # Sistema de ecuaciones
        self.eq_frame = ttk.LabelFrame(main_frame, text="Sistema de Ecuaciones (usar in1 e in2 para entradas)", padding="10")
        self.eq_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        
        example_text = "@(x1,x2,x3)[10*(x2-x1)+10*(3.705078*in2-18.220797-x1);\n"
        example_text += "x1*(28-x3)-x2+30*(3.705078*in2-18.220797-x1);\n"
        example_text += "x1*x2-(8/3)*x3+5*(3.705078*in2-18.220797-x1)]"
        
        self.eq_text = scrolledtext.ScrolledText(self.eq_frame, width=40, height=6)
        self.eq_text.insert(tk.END, example_text)
        self.eq_text.grid(row=0, column=0, sticky="nsew")
        
        # Configuración DAQ - Canal 1
        daq_frame1 = ttk.LabelFrame(main_frame, text="Configuración DAQ - Canal 1 (ao0)", padding="10")
        daq_frame1.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        
        ttk.Label(daq_frame1, text="Voltaje mínimo (V):").grid(row=0, column=0, sticky="w")
        self.voltage_min_entry1 = ttk.Entry(daq_frame1)
        self.voltage_min_entry1.insert(0, "-9.5")
        self.voltage_min_entry1.grid(row=0, column=1, sticky="w")
        
        ttk.Label(daq_frame1, text="Voltaje máximo (V):").grid(row=1, column=0, sticky="w")
        self.voltage_max_entry1 = ttk.Entry(daq_frame1)
        self.voltage_max_entry1.insert(0, "9.5")
        self.voltage_max_entry1.grid(row=1, column=1, sticky="w")
        
        ttk.Label(daq_frame1, text="Estado a enviar:").grid(row=2, column=0, sticky="w")
        self.state_entry1 = ttk.Entry(daq_frame1, width=5)
        self.state_entry1.insert(0, "1")
        self.state_entry1.grid(row=2, column=1, sticky="w")
        
        # Configuración DAQ - Canal 2
        daq_frame2 = ttk.LabelFrame(main_frame, text="Configuración DAQ - Canal 2 (ao1)", padding="10")
        daq_frame2.grid(row=5, column=0, sticky="ew", padx=5, pady=5)
        
        ttk.Label(daq_frame2, text="Voltaje mínimo (V):").grid(row=0, column=0, sticky="w")
        self.voltage_min_entry2 = ttk.Entry(daq_frame2)
        self.voltage_min_entry2.insert(0, "-9.5")
        self.voltage_min_entry2.grid(row=0, column=1, sticky="w")
        
        ttk.Label(daq_frame2, text="Voltaje máximo (V):").grid(row=1, column=0, sticky="w")
        self.voltage_max_entry2 = ttk.Entry(daq_frame2)
        self.voltage_max_entry2.insert(0, "9.5")
        self.voltage_max_entry2.grid(row=1, column=1, sticky="w")
        
        ttk.Label(daq_frame2, text="Estado a enviar:").grid(row=2, column=0, sticky="w")
        self.state_entry2 = ttk.Entry(daq_frame2, width=5)
        self.state_entry2.insert(0, "2")
        self.state_entry2.grid(row=2, column=1, sticky="w")
        
        # Ecuaciones de conversión
        conv_frame = ttk.LabelFrame(main_frame, text="Ecuaciones de Conversión", padding="10")
        conv_frame.grid(row=6, column=0, sticky="ew", padx=5, pady=5)

        # Canal 1
        ttk.Label(conv_frame, text="Canal 1 (ao0):", font=('Arial', 9, 'bold')).grid(row=0, column=0, sticky="w", columnspan=2)
        self.conversion_eq1_direct = ttk.Label(conv_frame, text="Voltaje = m * x + b")
        self.conversion_eq1_direct.grid(row=1, column=0, sticky="w", padx=10)
        self.conversion_eq1_inverse = ttk.Label(conv_frame, text="x = p * Voltaje + q")
        self.conversion_eq1_inverse.grid(row=1, column=1, sticky="w", padx=10)

        # Separador
        ttk.Separator(conv_frame, orient='horizontal').grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)

        # Canal 2
        ttk.Label(conv_frame, text="Canal 2 (ao1):", font=('Arial', 9, 'bold')).grid(row=3, column=0, sticky="w", columnspan=2)
        self.conversion_eq2_direct = ttk.Label(conv_frame, text="Voltaje = m * x + b")
        self.conversion_eq2_direct.grid(row=4, column=0, sticky="w", padx=10)
        self.conversion_eq2_inverse = ttk.Label(conv_frame, text="x = p * Voltaje + q")
        self.conversion_eq2_inverse.grid(row=4, column=1, sticky="w", padx=10)
        
        # Botones de control
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=7, column=0, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Iniciar Observador", command=self.start_observer)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Detener", command=self.stop_observer, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Configurar expansión
        main_frame.columnconfigure(0, weight=1)
    
    def parse_lambda_equation(self, text):
        """Convierte texto en formato lambda a función y detecta variables"""
        try:
            # Limpiar y extraer la expresión
            cleaned_text = re.sub(r'#.*', '', text).strip()
            if not cleaned_text.startswith('@(') or ']' not in cleaned_text:
                raise ValueError("Formato lambda incorrecto. Use: @(x1,x2,...)[expr1;expr2;...]")
            
            # Extraer variables
            vars_part = cleaned_text.split(')')[0][2:]
            variables = [v.strip() for v in vars_part.split(',') if v.strip()]
            
            if not variables:
                raise ValueError("No se detectaron variables en la función")
            
            # Verificar nombres de variables
            for i, var in enumerate(variables):
                if var != f'x{i+1}':
                    raise ValueError(f"Las variables deben llamarse x1, x2,... en orden. Encontrado: {var}")
            
            num_vars = len(variables)
            
            # Extraer expresiones
            expr_part = cleaned_text.split('[')[1].split(']')[0]
            expressions = [e.strip() for e in expr_part.split(';') if e.strip()]
            
            if len(expressions) != num_vars:
                raise ValueError(f"Número incorrecto de ecuaciones (esperado {num_vars}, obtenido {len(expressions)})")
            
            # Crear función del sistema que incluye in1 e in2
            def sistema(*args):
                if len(args) != num_vars:
                    raise ValueError(f"Se esperaban {num_vars} argumentos")
                
                vars_dict = {f'x{i+1}': args[i] for i in range(num_vars)}
                vars_dict['in1'] = self.in1
                vars_dict['in2'] = self.in2
                result = []
                
                for expr in expressions:
                    try:
                        # Reemplazar potencias y evaluar
                        expr = expr.replace('^', '**')
                        result.append(eval(expr, {'__builtins__': None}, vars_dict))
                    except Exception as e:
                        raise ValueError(f"Error evaluando expresión '{expr}': {str(e)}")
                
                return np.array(result)
            
            return sistema, num_vars
        
        except Exception as e:
            raise ValueError(f"Error procesando ecuación lambda: {str(e)}")
    
    def parse_matrix_input(self, text):
        """Convierte texto a matriz de límites"""
        try:
            cleaned_text = re.sub(r'#.*', '', text)
            limits = ast.literal_eval(cleaned_text.strip())
            
            if not isinstance(limits, list):
                raise ValueError("Debe ser una lista de listas")
                
            for i, item in enumerate(limits):
                if not isinstance(item, list) or len(item) != 2:
                    raise ValueError(f"Elemento {i+1} debe ser [min, max]")
                if item[0] >= item[1]:
                    raise ValueError(f"Para elemento {i+1}, min debe ser < max")
            
            return limits
        except Exception as e:
            raise ValueError(f"Formato de matriz inválido: {str(e)}")
    
    def update_conversion_equations(self, limits, state1, state2, v_min1, v_max1, v_min2, v_max2):
        """Actualiza las ecuaciones de conversión mostradas"""
        try:
            # Validar estados
            if state1 < 1 or state1 > len(limits) or state2 < 1 or state2 > len(limits):
                raise ValueError("Estado fuera de rango")
            
            # Obtener límites para cada estado
            min1, max1 = limits[state1-1]
            min2, max2 = limits[state2-1]
            
            # Calcular pendientes (m) para cada canal
            m1 = (v_max1 - v_min1) / (max1 - min1)
            b1 = v_min1 - m1 * min1
            
            m2 = (v_max2 - v_min2) / (max2 - min2)
            b2 = v_min2 - m2 * min2
            
            # Actualizar etiquetas con ecuaciones directas
            self.conversion_eq1_direct.config(text=f"Voltaje = {m1:.6f} * x{state1} + {b1:.6f}")
            self.conversion_eq2_direct.config(text=f"Voltaje = {m2:.6f} * x{state2} + {b2:.6f}")
            
            # Actualizar etiquetas con ecuaciones inversas
            if m1 != 0:
                inv_m1 = 1/m1
                inv_b1 = -b1/m1
                self.conversion_eq1_inverse.config(text=f"x{state1} = {inv_m1:.6f} * Voltaje + {inv_b1:.6f}")
            else:
                self.conversion_eq1_inverse.config(text="No aplicable (m=0)")
            
            if m2 != 0:
                inv_m2 = 1/m2
                inv_b2 = -b2/m2
                self.conversion_eq2_inverse.config(text=f"x{state2} = {inv_m2:.6f} * Voltaje + {inv_b2:.6f}")
            else:
                self.conversion_eq2_inverse.config(text="No aplicable (m=0)")
            
            return m1, b1, m2, b2
            
        except Exception as e:
            self.conversion_eq1_direct.config(text=f"Error: {str(e)}")
            self.conversion_eq1_inverse.config(text="")
            self.conversion_eq2_direct.config(text="")
            self.conversion_eq2_inverse.config(text="")
            return None, None, None, None
    
    def start_observer(self):
        if self.running:
            return
            
        try:
            # Obtener parámetros básicos
            h = float(self.h_entry.get())
            duration = float(self.duration_entry.get())
            
            # Procesar condiciones iniciales
            ci_str = [x.strip() for x in self.ci_entry.get().split(',') if x.strip()]
            ci = [float(valor) for valor in ci_str]
            
            # Procesar matriz de límites
            limits_text = self.limits_text.get("1.0", tk.END)
            limits = self.parse_matrix_input(limits_text)
            
            # Validar consistencia entre condiciones iniciales y límites
            if len(ci) != len(limits):
                raise ValueError(f"Condiciones iniciales ({len(ci)}) y límites ({len(limits)}) deben tener la misma dimensión")
            
            # Procesar ecuaciones lambda
            eq_text = self.eq_text.get("1.0", tk.END)
            sistema, num_vars = self.parse_lambda_equation(eq_text)
            
            # Validar consistencia con condiciones iniciales
            if len(ci) != num_vars:
                raise ValueError(f"Condiciones iniciales ({len(ci)}) y sistema ({num_vars}) deben tener la misma dimensión")
            
            # Configuración del Canal 1
            v_min1 = float(self.voltage_min_entry1.get())
            v_max1 = float(self.voltage_max_entry1.get())
            state1 = int(self.state_entry1.get())
            
            # Configuración del Canal 2
            v_min2 = float(self.voltage_min_entry2.get())
            v_max2 = float(self.voltage_max_entry2.get())
            state2 = int(self.state_entry2.get())
            
            # Validar estados seleccionados
            if state1 < 1 or state1 > num_vars or state2 < 1 or state2 > num_vars:
                raise ValueError(f"Los estados deben estar entre 1 y {num_vars}")
            
            # Validar parámetros
            if h <= 0 or duration <= 0:
                raise ValueError("El paso y la duración deben ser positivos")
            if v_min1 >= v_max1 or v_min2 >= v_max2:
                raise ValueError("El voltaje mínimo debe ser menor al máximo en ambos canales")
            
            # Actualizar ecuaciones de conversión
            m1, b1, m2, b2 = self.update_conversion_equations(
                limits, state1, state2, 
                v_min1, v_max1, v_min2, v_max2
            )
            
            if None in [m1, b1, m2, b2]:
                raise ValueError("Error en ecuaciones de conversión")
            
            # Configurar simulación
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            n_steps = int(duration / h)
            
            # Método RK4 genérico
            # IMPORTANTE: No indagué en el uso de un método numérico de mayor 
            # grado, aunque la lógica me dice que tal vez un RK2 puede manejar
            # pasos más pequeños. Siempre simula tu modelo para verificar cuál
            # método es más útil.
            def odesRK4step(f, ci, h):
                k1 = f(*ci)
                k2 = f(*(ci + 0.5 * h * k1))
                k3 = f(*(ci + 0.5 * h * k2))
                k4 = f(*(ci + h * k3))
                return ci + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            
            # Variables compartidas
            ci = np.array(ci)
            barrier = threading.Barrier(3)
            datos_listos = threading.Event()
            
            # Hilo Reloj
            def reloj():
                t0 = time.time()
                for i in range(n_steps):
                    if not self.running:
                        break
                    next_tick = t0 + i * h
                    time.sleep(max(0, next_tick - time.time()))
                    barrier.wait()
            
            # Hilo Cálculo
            def calculo():
                nonlocal ci
                for _ in range(n_steps):
                    if not self.running:
                        break
                    barrier.wait()
                    ci = odesRK4step(sistema, ci, h)
                    datos_listos.set()
            
            # Hilo DAQ (lectura y escritura)
            def daq_operations():
                with nidaqmx.Task() as read_task, nidaqmx.Task() as write_task:
                    # Configurar canales de entrada (lectura)
                    read_task.ai_channels.add_ai_voltage_chan("Dev1/ai0", min_val=-10.0, max_val=10.0)
                    read_task.ai_channels.add_ai_voltage_chan("Dev1/ai1", min_val=-10.0, max_val=10.0)
                    
                    # Configurar canales de salida (escritura)
                    write_task.ao_channels.add_ao_voltage_chan("Dev1/ao0", min_val=-10.0, max_val=10.0)
                    write_task.ao_channels.add_ao_voltage_chan("Dev1/ao1", min_val=-10.0, max_val=10.0)
                    
                    for _ in range(n_steps):
                        if not self.running:
                            break
                        barrier.wait()
                        
                        try:
                            # 1. Leer entradas analógicas
                            inputs = read_task.read()
                            self.in1 = float(inputs[0])
                            self.in2 = float(inputs[1])
                            
                            # Esperar a que cálculo termine
                            datos_listos.wait()
                            
                            # 2. Convertir estados seleccionados a voltaje
                            # Para canal 1
                            valor_estado1 = ci[state1-1]
                            voltaje1 = m1 * (valor_estado1 - limits[state1-1][0]) + v_min1
                            voltaje1 = np.clip(voltaje1, -10.0, 10.0)  # Limitar a rango DAQ
                            
                            # Para canal 2
                            valor_estado2 = ci[state2-1]
                            voltaje2 = m2 * (valor_estado2 - limits[state2-1][0]) + v_min2
                            voltaje2 = np.clip(voltaje2, -10.0, 10.0)  # Limitar a rango DAQ
                            
                            # 3. Escribir salidas
                            write_task.write([voltaje1, voltaje2])
                            
                        except Exception as e:
                            print(f"Error DAQ: {e}")
                            # En caso de error, enviar 0V a ambos canales
                            write_task.write([0.0, 0.0])
                        finally:
                            datos_listos.clear()
            
            # Iniciar hilos
            self.threads = [
                threading.Thread(target=reloj),
                threading.Thread(target=calculo),
                threading.Thread(target=daq_operations)
            ]
            
            for thread in self.threads:
                thread.start()
                
            messagebox.showinfo("Observador", "Observador en ejecución...")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al iniciar: {str(e)}")
            self.stop_observer()
    
    def stop_observer(self):
        if not self.running:
            return
            
        self.running = False
        
        for thread in self.threads:
            if thread.is_alive():
                thread.join()
        
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        messagebox.showinfo("Observador", "Observador detenido")
    
    def on_closing(self):
        self.stop_observer()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ObserverApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()