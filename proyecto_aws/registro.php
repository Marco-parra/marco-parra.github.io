<?php
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
    $email = $_POST['email'];

    // Hashear la contraseña antes de guardarla en la base de datos
    $hashed_password = password_hash($password, PASSWORD_DEFAULT);

    // Preparar la consulta SQL para insertar los datos en la base de datos
    $stmt = $conexion->prepare('INSERT INTO usuarios (username, password, email) VALUES (?, ?, ?)');
    $stmt->bind_param('sss', $username, $hashed_password, $email);

    // Ejecutar la consulta
    if ($stmt->execute()) {
        // Registro exitoso, redireccionar a una página de éxito o mostrar un mensaje de éxito
        header('Location: index.html');
        exit();
    } else {
        // Error al registrar, redireccionar a una página de error o mostrar un mensaje de error
        header('Location: error_registro.html');
        exit();
    }

    // Cerrar la conexión y liberar los recursos
    $stmt->close();
    $conexion->close();
} else {
    // Si se accede directamente a este archivo sin enviar datos por POST, redireccionar a la página de registro
    header('Location: index.html');
}
?>
