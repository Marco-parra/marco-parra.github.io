<?php
session_start();

if (!isset($_SESSION['loggedin'])) {
    header('Location: index.html');
    exit;
}

// Conectar a la base de datos
$DATABASE_HOST = '18.224.240.3';
$DATABASE_USER = 'marco';
$DATABASE_PASS = '1234';
$DATABASE_NAME = 'seguridad';

$conexion = new mysqli($DATABASE_HOST, $DATABASE_USER, $DATABASE_PASS, $DATABASE_NAME);

if ($conexion->connect_error) {
    exit('Fallo en la conexión de MySQL: ' . $conexion->connect_error);
}

function obtenerCasas($conexion) {
    $sql = "SELECT * FROM casas";
    $resultado = $conexion->query($sql);
    return $resultado->fetch_all(MYSQLI_ASSOC);
}

if ($_SERVER['REQUEST_METHOD'] == 'POST') {
    if (isset($_POST['crear'])) {
        $nombre = $_POST['nombre'];
        $imagen = $_POST['imagen'];
        $lema = $_POST['lema'];
        $blason = $_POST['blason'];

        $sql = "INSERT INTO casas (nombre, imagen, lema, blason) VALUES (?, ?, ?, ?)";
        $stmt = $conexion->prepare($sql);
        $stmt->bind_param("ssss", $nombre, $imagen, $lema, $blason);
        $stmt->execute();
        $stmt->close();
    } elseif (isset($_POST['actualizar'])) {
        $id = $_POST['id'];
        $nombre = $_POST['nombre'];
        $imagen = $_POST['imagen'];
        $lema = $_POST['lema'];
        $blason = $_POST['blason'];

        $sql = "UPDATE casas SET nombre = ?, imagen = ?, lema = ?, blason = ? WHERE id = ?";
        $stmt = $conexion->prepare($sql);
        $stmt->bind_param("ssssi", $nombre, $imagen, $lema, $blason, $id);
        $stmt->execute();
        $stmt->close();
    } elseif (isset($_POST['eliminar'])) {
        $id = $_POST['id'];

        $sql = "DELETE FROM casas WHERE id = ?";
        $stmt = $conexion->prepare($sql);
        $stmt->bind_param("i", $id);
        $stmt->execute();
        $stmt->close();
    }
}

$casas = obtenerCasas($conexion);
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CRUD Casas de Juego de Tronos</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #1540b683;
            margin: 0;
            padding: 0;
        }
        .container {
            max-width: 800px;
            margin: 20px auto;
            padding: 20px;
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        table, th, td {
            border: 1px solid #ddd;
        }
        th, td {
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        form {
            display: inline;
        }
        input[type="text"], input[type="url"] {
            width: 100%;
            padding: 5px;
            margin: 5px 0;
            box-sizing: border-box;
        }
        button {
            padding: 5px 10px;
            background-color: #007bff;
            color: #fff;
            border: none;
            border-radius: 3px;
            cursor: pointer;
        }
        button:hover {
            background-color: #0056b3;
        }
        img {
            max-width: 50px;
            max-height: 50px;
        }
        .acciones button {
            margin: 2px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>CRUD Casas de Juego de Tronos</h1>

        <h2>Crear Casa</h2>
        <form action="crud.php" method="post">
            <label for="nombre">Nombre:</label>
            <input type="text" name="nombre" id="nombre" required>
            <label for="imagen">Imagen URL:</label>
            <input type="url" name="imagen" id="imagen" required>
            <label for="lema">Lema:</label>
            <input type="text" name="lema" id="lema" required>
            <label for="blason">Blasón:</label>
            <input type="text" name="blason" id="blason" required>
            <button type="submit" name="crear">Crear</button>
        </form>

        <h2>Casas Registradas</h2>
        <table>
            <tr>
                <th>ID</th>
                <th>Nombre</th>
                <th>Imagen</th>
                <th>Lema</th>
                <th>Blasón</th>
                <th>Acciones</th>
            </tr>
            <?php foreach ($casas as $casa): ?>
                <tr>
                    <td><?= $casa['id'] ?></td>
                    <td><?= $casa['nombre'] ?></td>
                    <td><img src="<?= $casa['imagen'] ?>" alt="Emblema"></td>
                    <td><?= $casa['lema'] ?></td>
                    <td><?= $casa['blason'] ?></td>
                    <td class="acciones">
                        <form action="crud.php" method="post" style="display:inline;">
                            <input type="hidden" name="id" value="<?= $casa['id'] ?>">
                            <button type="submit" name="eliminar">Eliminar</button>
                        </form>
                        <form action="crud.php" method="post" style="display:inline;">
                            <input type="hidden" name="id" value="<?= $casa['id'] ?>">
                            <button type="button" onclick="editarCasa(<?= $casa['id'] ?>)">Editar</button>
                        </form>
                        <form action="detalles.php" method="get" style="display:inline;">
                            <input type="hidden" name="id" value="<?= $casa['id'] ?>">
                            <button type="submit">Ver Detalles</button>
                        </form>
                    </td>
                </tr>
                <tr id="editar-<?= $casa['id'] ?>" style="display:none;">
                    <td colspan="6">
                        <form action="crud.php" method="post">
                            <input type="hidden" name="id" value="<?= $casa['id'] ?>">
                            <label for="nombre-<?= $casa['id'] ?>">Nombre:</label>
                            <input type="text" name="nombre" id="nombre-<?= $casa['id'] ?>" value="<?= $casa['nombre'] ?>" required>
                            <label for="imagen-<?= $casa['id'] ?>">Imagen URL:</label>
                            <input type="url" name="imagen" id="imagen-<?= $casa['id'] ?>" value="<?= $casa['imagen'] ?>" required>
                            <label for="lema-<?= $casa['id'] ?>">Lema:</label>
                            <input type="text" name="lema" id="lema-<?= $casa['id'] ?>" value="<?= $casa['lema'] ?>" required>
                            <label for="blason-<?= $casa['id'] ?>">Blasón:</label>
                            <input type="text" name="blason" id="blason-<?= $casa['id'] ?>" value="<?= $casa['blason'] ?>" required>
                            <button type="submit" name="actualizar">Actualizar</button>
                        </form>
                    </td>
                </tr>
            <?php endforeach; ?>
        </table>
    </div>
    <script>
        function editarCasa(id) {
            var fila = document.getElementById('editar-' + id);
            if (fila.style.display === 'none') {
                fila.style.display = '';
            } else {
                fila.style.display = 'none';
            }
        }
    </script>
</body>
</html>
