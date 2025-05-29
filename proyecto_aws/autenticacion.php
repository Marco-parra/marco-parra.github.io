<?php
session_start();

// Verificar si se han enviado datos del formulario
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    // Establecer conexión a la base de datos
    $DATABASE_HOST = '18.224.240.3';
    $DATABASE_USER = 'marco';
    $DATABASE_PASS = '1234';
    $DATABASE_NAME = 'seguridad';
    $conexion = mysqli_connect($DATABASE_HOST, $DATABASE_USER, $DATABASE_PASS, $DATABASE_NAME);
    if (mysqli_connect_error()) {
        exit('Fallo en la conexión de MySQL:' . mysqli_connect_error());
    }

    // Obtener los datos del formulario
    $username = $_POST['username'];
    $password = $_POST['password'];

    // Preparar la consulta SQL para obtener la contraseña hasheada del usuario
    $stmt = $conexion->prepare('SELECT id, password FROM usuarios WHERE username = ?');
    $stmt->bind_param('s', $username);
    $stmt->execute();
    $stmt->store_result();

    // Verificar si se encontró un usuario con el nombre proporcionado
    if ($stmt->num_rows > 0) {
        $stmt->bind_result($id, $hashed_password);
        $stmt->fetch();

        // Verificar si la contraseña proporcionada coincide con la contraseña hasheada en la base de datos
        if (password_verify($password, $hashed_password)) {
            // Inicio de sesión exitoso, se crea la sesión
            session_regenerate_id();
            $_SESSION['loggedin'] = TRUE;
            $_SESSION['name'] = $username;
            $_SESSION['id'] = $id;
            // Redireccionar a la página de inicio después de iniciar sesión
            header('Location: inicio.php');
            exit();
        } else {
            // Contraseña incorrecta, redireccionar a la página de inicio de sesión con un mensaje de error
            header('Location: index.html?error=password');
            exit();
        }
    } else {
        // Usuario no encontrado, redireccionar a la página de inicio de sesión con un mensaje de error
        header('Location: index.html?error=user');
        exit();
    }

    // Cerrar la conexión y liberar los recursos
    $stmt->close();
    $conexion->close();
} else {
    // Si se accede directamente a este archivo sin enviar datos por POST, redireccionar a la página de inicio de sesión
    header('Location: index.html');
    exit();
}
?>
