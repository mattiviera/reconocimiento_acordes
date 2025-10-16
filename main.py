from src.train import train_model
from src.predict import predict_chord

def main():
    while True:
        print("\n" + "="*40)
        print("1. Entrenar modelo")
        print("2. Predecir acorde en tiempo real")
        print("3. Salir")
        print("="*40)
        option = input("Selecciona una opción (1, 2 o 3): ")

        if option == "1":
            train_model()
        elif option == "2":
            predict_chord()
        elif option == "3":
            print("Saliendo...")
            break
        else:
            print("Opción inválida. Inténtalo de nuevo.")

if __name__ == "__main__":
    main()