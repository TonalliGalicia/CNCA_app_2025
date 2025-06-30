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


class MultiChannelDAQApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulador de EDOs y Generador de Señales DAQ Multicanal")
        
        # Variables compartidas
        self.shared_limits = None
        self.shared_solution = None
        self.shared_time = None
        self.shared_ci = None
        self.shared_system = None
        
        # Variables de control para DAQ
        self.running = False
        self.threads = []
        
        # Crear paneles divididos
        self.create_left_panel()  # Simulador de EDOs
        self.create_right_panel()  # Generador de Señales DAQ multicanal
        
        # Separador visual
        separator = ttk.Separator(root, orient='vertical')
        separator.pack(side='left', fill='y', padx=5)
    
    def create_left_panel(self):
        # Frame para el simulador de EDOs (izquierda)
        self.left_frame = ttk.Frame(self.root, padding="10")
        self.left_frame.pack(side='left', fill='both', expand=True)
        
        # Variables para almacenar los parámetros
        self.system_eq = tk.StringVar(value="@(x1,x2,x3)[10*(x2-x1);x1*(28-x3)-x2;x1*x2-8/3*x3]")
        self.step_size = tk.StringVar(value="0.001")
        self.initial_cond = tk.StringVar(value="14,25,20")
        self.max_time = tk.StringVar(value="50")
        self.num_samples = tk.StringVar(value="4000")
        self.state_to_plot = tk.StringVar(value="1")
        
        # Resultados
        self.time = None
        self.solution = None
        self.limits = None
        
        # Crear widgets para el panel izquierdo
        self.create_ode_widgets()
    
    def create_right_panel(self):
        # Frame para el generador de señales DAQ (derecha)
        self.right_frame = ttk.Frame(self.root, padding="10")
        self.right_frame.pack(side='right', fill='both', expand=True)
        
        # Crear widgets para el panel derecho
        self.create_daq_widgets()
    
    def create_ode_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self.left_frame)
        main_frame.pack(fill='both', expand=True)
        
        # Sección de entrada de sistema de ecuaciones
        ttk.Label(main_frame, text="Sistema de EDOs (usar x1, x2, x3, etc.):").grid(row=0, column=0, sticky=tk.W, pady=(0,5))
        self.system_entry = ttk.Entry(main_frame, textvariable=self.system_eq, width=40)
        self.system_entry.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0,10))
        
        # Parámetros de simulación
        ttk.Label(main_frame, text="Paso (h):").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(main_frame, textvariable=self.step_size, width=15).grid(row=2, column=1, sticky=tk.W)
        
        ttk.Label(main_frame, text="Condiciones iniciales (separadas por comas):").grid(row=3, column=0, sticky=tk.W)
        ttk.Entry(main_frame, textvariable=self.initial_cond, width=15).grid(row=3, column=1, sticky=tk.W)
        
        ttk.Label(main_frame, text="Tiempo máximo de simulación:").grid(row=4, column=0, sticky=tk.W)
        ttk.Entry(main_frame, textvariable=self.max_time, width=15).grid(row=4, column=1, sticky=tk.W)
        
        ttk.Label(main_frame, text="Número de muestras:").grid(row=5, column=0, sticky=tk.W)
        ttk.Entry(main_frame, textvariable=self.num_samples, width=15).grid(row=5, column=1, sticky=tk.W)
        
        # Botones de ejecución
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Simular", command=self.run_simulation).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Enviar a DAQ", command=self.send_to_daq).pack(side=tk.LEFT, padx=5)
        
        # Sección de resultados
        ttk.Label(main_frame, text="Límites de los estados:").grid(row=7, column=0, sticky=tk.W)
        self.results_text = tk.Text(main_frame, height=5, width=40, state=tk.DISABLED)
        self.results_text.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Entrada para estado a graficar
        ttk.Label(main_frame, text="Estado a graficar (ej. 1, 2, 3...):").grid(row=9, column=0, sticky=tk.W)
        self.state_entry = ttk.Entry(main_frame, textvariable=self.state_to_plot, width=5)
        self.state_entry.grid(row=9, column=1, sticky=tk.W)
        ttk.Button(main_frame, text="Graficar", command=self.plot_state).grid(row=10, column=0, columnspan=2, pady=5)
        
        # Gráfico
        self.figure, self.ax = plt.subplots(figsize=(6, 3))
        self.canvas = FigureCanvasTkAgg(self.figure, master=main_frame)
        self.canvas.get_tk_widget().grid(row=11, column=0, columnspan=2, pady=10)
        
        # Configurar expansión
        for i in range(12):
            main_frame.rowconfigure(i, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
    
    def create_daq_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self.right_frame)
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
        self.duration_entry.insert(0, "15")
        self.duration_entry.grid(row=1, column=1, sticky="w")
        
        # Condiciones iniciales
        self.ci_frame = ttk.LabelFrame(main_frame, text="Condiciones Iniciales", padding="10")
        self.ci_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        ttk.Label(self.ci_frame, text="Valores (separados por comas):").grid(row=0, column=0, sticky="w")
        self.ci_entry = ttk.Entry(self.ci_frame)
        self.ci_entry.insert(0, "")
        self.ci_entry.grid(row=0, column=1, sticky="ew")
        
        # Límites de estados (solo visualización)
        self.limits_frame = ttk.LabelFrame(main_frame, text="Límites de Estados (de la simulación)", padding="10")
        self.limits_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        self.limits_text = scrolledtext.ScrolledText(self.limits_frame, width=40, height=4, state=tk.DISABLED)
        self.limits_text.grid(row=0, column=0, sticky="nsew")
        
        # Sistema de ecuaciones
        self.eq_frame = ttk.LabelFrame(main_frame, text="Sistema de Ecuaciones", padding="10")
        self.eq_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        
        example_text = "# Ejemplo para sistema de Lorenz:\n"
        example_text += "@(x1,x2,x3)[10*(x2-x1); x1*(28-x3)-x2; x1*x2-(8/3)*x3]"
        
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
        
        self.state_label1 = ttk.Label(daq_frame1, text="Estado a enviar:")
        self.state_label1.grid(row=2, column=0, sticky="w")
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
        
        self.state_label2 = ttk.Label(daq_frame2, text="Estado a enviar:")
        self.state_label2.grid(row=2, column=0, sticky="w")
        self.state_entry2 = ttk.Entry(daq_frame2, width=5)
        self.state_entry2.insert(0, "2")
        self.state_entry2.grid(row=2, column=1, sticky="w")
        
        # En el método create_daq_widgets(), reemplazar el conv_frame con:

            # Ecuaciones de conversión para ambos canales
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
    
        # Ajustar columnas
        conv_frame.columnconfigure(0, weight=1)
        conv_frame.columnconfigure(1, weight=1)
        
        # Botones de control
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=7, column=0, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Iniciar Simulación DAQ", command=self.start_daq_simulation)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Detener", command=self.stop_simulation, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Configurar expansión
        main_frame.columnconfigure(0, weight=1)
    
    # ==============================================
    # Métodos para la simulación de EDOs
    # ==============================================
    
    def run_simulation(self):
        try:
            # Obtener parámetros
            system_str = self.system_eq.get()
            h = float(self.step_size.get())
            ci = np.array([float(x.strip()) for x in self.initial_cond.get().split(',')])
            tmax = float(self.max_time.get())
            nm = int(self.num_samples.get())
            
            # Validar parámetros
            if h <= 0 or tmax <= 0 or nm <= 0:
                raise ValueError("Los parámetros deben ser positivos")
                
            # Convertir el string del sistema a una función lambda
            system_lambda = self.parse_system(system_str)
            
            # Ejecutar RK4 modificado que devuelve también los límites
            self.time, self.solution, self.limits = self.odesRK4m_min_max(system_lambda, ci, tmax, h, nm)
            
            # Mostrar resultados
            self.show_results()
            
            # Graficar el primer estado por defecto
            self.plot_state()
            
            # Guardar datos para compartir con DAQ
            self.shared_limits = self.limits
            self.shared_solution = self.solution
            self.shared_time = self.time
            self.shared_ci = ci
            self.shared_system = system_str
            
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")
    
    def parse_system(self, system_str):
        try:
            var_part = system_str.split('[')[0]
            vars_str = var_part.split('(')[1].split(')')[0]
            variables = [v.strip() for v in vars_str.split(',')]
            
            eq_part = system_str.split('[')[1].split(']')[0]
            equations = [eq.strip() for eq in eq_part.split(';')]
            
            lambda_str = f"lambda {','.join(variables)}: ["
            lambda_str += ','.join(equations)
            lambda_str += "]"
            
            return eval(lambda_str)
        except Exception as e:
            raise ValueError(f"Formato de sistema no válido: {str(e)}")
    
    def odesRK4m_min_max(self, vfa, ci, tmax, tstep, nm):
        n = len(ci)
        d = self.red_inf(tmax/tstep) + 1
        
        if nm > d:
            raise ValueError("El número de muestras solicitado es mayor al número de puntos generados")
        
        tm = tmax/(nm-1)
        M = np.zeros((n, nm))
        M[:, 0] = ci
        tout = np.zeros(nm)
        limites = np.zeros((n, 2))
        
        # Inicializar límites con las condiciones iniciales
        for i in range(n):
            limites[i, 0] = ci[i]
            limites[i, 1] = ci[i]
        
        K1 = np.zeros(n)
        K2 = np.zeros(n)
        K3 = np.zeros(n)
        K4 = np.zeros(n)
        nx = np.zeros(n)
        nxp = ci.copy()
        pos = 1
        tini = 0.0
        
        for j in range(1, d):
            nx = nxp.copy()
            K1 = np.array(vfa(*nx))
            nx = nxp + 0.5 * tstep * K1
            K2 = np.array(vfa(*nx))
            nx = nxp + 0.5 * tstep * K2
            K3 = np.array(vfa(*nx))
            nx = nxp + tstep * K3
            K4 = np.array(vfa(*nx))
            nxp = nxp + (tstep/6.0) * (K1 + 2*K2 + 2*K3 + K4)
            tini += tstep
            
            # Actualizar límites
            for i in range(n):
                if nxp[i] < limites[i, 0]:
                    limites[i, 0] = nxp[i]
                if nxp[i] > limites[i, 1]:
                    limites[i, 1] = nxp[i]
            
            if self.red_inf(tini/(tm*pos)) == 1:
                pos += 1
                M[:, pos-1] = nxp
                tout[pos-1] = tini
        
        return tout, M, limites
    
    def red_inf(self, n):
        if round(n) - n > 1e-10:
            return round(n) - 1
        else:
            return round(n)
    
    def show_results(self):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        if self.limits is not None:
            # Encabezado con formato mejorado
            header = "Estado".ljust(10) + "Mínimo".rjust(15) + "Máximo".rjust(15)
            separator = "-" * len(header)
            
            self.results_text.insert(tk.END, header + "\n")
            self.results_text.insert(tk.END, separator + "\n")
            
            # Mostrar cada estado con su mínimo y máximo
            for i in range(len(self.limits)):
                estado = f"x{i+1}".ljust(10)
                minimo = f"{self.limits[i,0]:.6f}".rjust(15)
                maximo = f"{self.limits[i,1]:.6f}".rjust(15)
                self.results_text.insert(tk.END, f"{estado}{minimo}{maximo}\n")
        
        self.results_text.config(state=tk.DISABLED)
    
    def plot_state(self):
        if self.time is not None and self.solution is not None:
            try:
                state_idx = int(self.state_to_plot.get()) - 1
                num_states = self.solution.shape[0]
                
                if 0 <= state_idx < num_states:
                    self.ax.clear()
                    self.ax.plot(self.time, self.solution[state_idx, :], 'b-', linewidth=1.5)
                    self.ax.set_title(f"Evolución del estado x{state_idx+1}", fontsize=10)
                    self.ax.set_xlabel("Tiempo", fontsize=8)
                    self.ax.set_ylabel(f"x{state_idx+1}(t)", fontsize=8)
                    self.ax.grid(True, linestyle='--', alpha=0.7)
                    self.canvas.draw()
                else:
                    messagebox.showerror("Error", f"El estado debe estar entre 1 y {num_states}")
            except ValueError:
                messagebox.showerror("Error", "Ingrese un número válido para el estado")
        else:
            messagebox.showerror("Error", "Primero debe ejecutar una simulación")
    
    def send_to_daq(self):
        """Envía los datos de la simulación actual al panel DAQ"""
        if self.time is None or self.solution is None or self.limits is None:
            messagebox.showerror("Error", "Primero debe ejecutar una simulación")
            return
        
        # Actualizar el panel DAQ con los datos de la simulación
        self.eq_text.delete("1.0", tk.END)
        self.eq_text.insert(tk.END, self.system_eq.get())
        
        self.ci_entry.delete(0, tk.END)
        self.ci_entry.insert(0, self.initial_cond.get())
        
        # Actualizar los límites (solo visualización)
        self.limits_text.config(state=tk.NORMAL)
        self.limits_text.delete("1.0", tk.END)
        
        if self.limits is not None:
            for i in range(len(self.limits)):
                self.limits_text.insert(tk.END, f"x{i+1}: [{self.limits[i,0]:.6f}, {self.limits[i,1]:.6f}]\n")
        
        self.limits_text.config(state=tk.DISABLED)
        
        # Actualizar ecuaciones de conversión para ambos canales
        try:
            state1 = int(self.state_entry1.get())
            if 1 <= state1 <= len(self.limits):
                self.update_conversion_equations(
                    self.limits, 
                    [float(self.voltage_min_entry1.get()), float(self.voltage_max_entry1.get())], 
                    state1,
                    channel=1
                )
            
            state2 = int(self.state_entry2.get())
            if 1 <= state2 <= len(self.limits):
                self.update_conversion_equations(
                    self.limits, 
                    [float(self.voltage_min_entry2.get()), float(self.voltage_max_entry2.get())], 
                    state2,
                    channel=2
                )
        except:
            pass
    
    # ==============================================
    # Métodos para la generación de señales DAQ
    # ==============================================
    
    def update_conversion_equations(self, limits, voltages, state, channel=1):
        """Actualiza las ecuaciones de conversión mostradas para el canal especificado"""
        try:
            min_edo, max_edo = limits[state-1]
            v_min, v_max = voltages
        
            # Calcular coeficientes para voltaje = m*valor + b (conversión directa)
            m = (v_max - v_min) / (max_edo - min_edo)
            b = v_min - m * min_edo
        
            # Calcular coeficientes inversos para valor = p*voltaje + q (conversión inversa)
            p = 1/m if m != 0 else 0
            q = -b/m if m != 0 else 0
        
            # Actualizar etiquetas con valores numéricos según el canal
            if channel == 1:
                # Ecuaciones para el Canal 1 (ao0)
                self.conversion_eq1_direct.config(text=f"Voltaje Canal 1 = {m:.6f} * x{state} + {b:.6f}")
                self.conversion_eq1_inverse.config(text=f"x{state} = {p:.6f} * Voltaje Canal 1 + {q:.6f}")
            else:
                    # Ecuaciones para el Canal 2 (ao1)
                    self.conversion_eq2_direct.config(text=f"Voltaje Canal 2 = {m:.6f} * x{state} + {b:.6f}")
                    self.conversion_eq2_inverse.config(text=f"x{state} = {p:.6f} * Voltaje Canal 2 + {q:.6f}")
        
        except Exception as e:
            error_msg = "Error calculando coeficientes"
            if channel == 1:
                self.conversion_eq1_direct.config(text=error_msg)
                self.conversion_eq1_inverse.config(text=error_msg)
            else:
                    self.conversion_eq2_direct.config(text=error_msg)
                    self.conversion_eq2_inverse.config(text=error_msg)
                    print(f"Error actualizando ecuaciones: {e}")

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
            
            # Crear función del sistema
            def sistema(*args):
                if len(args) != num_vars:
                    raise ValueError(f"Se esperaban {num_vars} argumentos")
                
                vars_dict = {f'x{i+1}': args[i] for i in range(num_vars)}
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
    
    def parse_matrix_input(self, text, num_vars):
        """Convierte texto a matriz de límites"""
        try:
            cleaned_text = re.sub(r'#.*', '', text)
            limits = ast.literal_eval(cleaned_text.strip())
            
            if not isinstance(limits, list) or len(limits) != num_vars:
                raise ValueError(f"Debe ser una lista con {num_vars} elementos")
                
            for i, item in enumerate(limits):
                if not isinstance(item, list) or len(item) != 2:
                    raise ValueError(f"Elemento {i+1} debe ser [min, max]")
                if item[0] >= item[1]:
                    raise ValueError(f"Para x{i+1}, min debe ser < max")
            
            return limits
        except Exception as e:
            raise ValueError(f"Formato de matriz inválido: {str(e)}")
    
    def start_daq_simulation(self):
        if self.running:
            return
            
        try:
            # Obtener parámetros básicos
            h = float(self.h_entry.get())
            duration = float(self.duration_entry.get())
            
            # Procesar ecuaciones lambda (esto detectará el número de variables)
            eq_text = self.eq_text.get("1.0", tk.END)
            sistema, num_vars = self.parse_lambda_equation(eq_text)
            
            # Actualizar etiquetas con el número detectado de variables
            self.ci_frame.config(text=f"Condiciones Iniciales (x1..x{num_vars})")
            
            # Procesar condiciones iniciales
            ci_str = [x.strip() for x in self.ci_entry.get().split(',') if x.strip()]
            if len(ci_str) != num_vars:
                raise ValueError(f"Debe ingresar {num_vars} valores separados por comas")
            ci = [float(valor) for valor in ci_str]
            
            # Usar los límites compartidos si están disponibles
            if self.shared_limits is not None and len(self.shared_limits) == num_vars:
                limits = self.shared_limits
            else:
                # Si no hay límites compartidos, intentar obtenerlos del texto
                limits_text = self.limits_text.get("1.0", tk.END)
                if limits_text.strip():
                    limits = []
                    for line in limits_text.split('\n'):
                        if ':' in line:
                            lim_part = line.split(':')[1].strip().strip('[]')
                            limits.append([float(x.strip()) for x in lim_part.split(',')])
                else:
                    raise ValueError("No se han definido límites para los estados")
            
            # Configuración del Canal 1
            voltages1 = [float(self.voltage_min_entry1.get()), 
                        float(self.voltage_max_entry1.get())]
            state1 = int(self.state_entry1.get())
            if state1 < 1 or state1 > num_vars:
                raise ValueError(f"El estado para Canal 1 debe ser entre 1 y {num_vars}")
            
            # Configuración del Canal 2
            voltages2 = [float(self.voltage_min_entry2.get()), 
                        float(self.voltage_max_entry2.get())]
            state2 = int(self.state_entry2.get())
            if state2 < 1 or state2 > num_vars:
                raise ValueError(f"El estado para Canal 2 debe ser entre 1 y {num_vars}")
            
            # Validar que los estados sean diferentes
            if state1 == state2:
                raise ValueError("Los estados para los canales 1 y 2 deben ser diferentes")
            
            # Validar parámetros
            if h <= 0 or duration <= 0:
                raise ValueError("El paso y la duración deben ser positivos")
            if voltages1[0] >= voltages1[1] or voltages2[0] >= voltages2[1]:
                raise ValueError("El voltaje mínimo debe ser menor al máximo en ambos canales")
            
            # Actualizar ecuaciones de conversión
            self.update_conversion_equations(limits, voltages1, state1, channel=1)
            self.update_conversion_equations(limits, voltages2, state2, channel=2)
            
            # Configurar simulación
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            n_steps = int(duration / h)
            
            # Método RK4 genérico
            def odesRK4step(f, ci, h):
                k1 = f(*ci)
                k2 = f(*(ci + 0.5 * h * k1))
                k3 = f(*(ci + 0.5 * h * k2))
                k4 = f(*(ci + h * k3))
                return ci + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            
            # Funciones de conversión a voltaje para ambos canales
            def convertir_a_voltaje(valor, edo, v_min, v_max, min_edo, max_edo):
                m = (v_max - v_min) / (max_edo - min_edo)
                b = v_min - m * min_edo
                voltaje = m * valor + b
                return round(np.clip(voltaje, v_min, v_max), 4)
            
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
            
            # Hilo DAQ (ahora maneja ambos canales)
            def escritura():
                with nidaqmx.Task() as task:
                    # Configurar ambos canales analógicos de salida
                    task.ao_channels.add_ao_voltage_chan(
                        "Dev1/ao0", 
                        min_val=voltages1[0], 
                        max_val=voltages1[1]
                    )
                    task.ao_channels.add_ao_voltage_chan(
                        "Dev1/ao1", 
                        min_val=voltages2[0], 
                        max_val=voltages2[1]
                    )
                    
                    for _ in range(n_steps):
                        if not self.running:
                            break
                        barrier.wait()
                        datos_listos.wait()
                        try:
                            # Calcular voltajes para ambos canales
                            min_edo1, max_edo1 = limits[state1-1]
                            voltaje1 = convertir_a_voltaje(
                                ci[state1-1], state1, 
                                voltages1[0], voltages1[1],
                                min_edo1, max_edo1
                            )
                            
                            min_edo2, max_edo2 = limits[state2-1]
                            voltaje2 = convertir_a_voltaje(
                                ci[state2-1], state2, 
                                voltages2[0], voltages2[1],
                                min_edo2, max_edo2
                            )
                            
                            # Escribir ambos canales simultáneamente
                            task.write([voltaje1, voltaje2])
                        except Exception as e:
                            print(f"Error DAQ: {e}")
                            task.write([0.0, 0.0])
                        datos_listos.clear()
            
            # Iniciar hilos
            self.threads = [
                threading.Thread(target=reloj),
                threading.Thread(target=calculo),
                threading.Thread(target=escritura)
            ]
            
            for thread in self.threads:
                thread.start()
                
            messagebox.showinfo("Simulación", "Enviando señales a los canales DAQ...")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al iniciar: {str(e)}")
            self.stop_simulation()
    
    def stop_simulation(self):
        if not self.running:
            return
            
        self.running = False
        
        for thread in self.threads:
            if thread.is_alive():
                thread.join()
        
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        messagebox.showinfo("Simulación", "Señales detenidas")
    
    def on_closing(self):
        self.stop_simulation()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MultiChannelDAQApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()