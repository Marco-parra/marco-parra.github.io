<?php
    include "conexion.php"; // Includes the PHP file containing database connection details

    // Check if form is submitted
    if ($_SERVER["REQUEST_METHOD"] == "POST") {
        // Collect data from form inputs
        $nombre = $_POST['nombre'];
        $apellido = $_POST['apellido'];
        $edad = $_POST['edad'];

        // Insert data into database
        $sql_insert = "INSERT INTO persona (nombre, apellido, edad) VALUES ('$nombre', '$apellido', $edad)";
        if (mysqli_query($conn, $sql_insert)) {
            echo '<script>Swal.fire("Saved!", "", "success");</script>'; // Display success message
        } else {
            echo '<script>Swal.fire("Error!", "Unable to save changes", "error");</script>'; // Display error message
        }
    }
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">
    <link rel="stylesheet" href="style.css">
</head>
<body class="bg-secondary-subtle">
    <div class="container mt-5">
        <div class="row">
            <div class="col">
                <div class="bg-success-subtle border border-success border-3 p-3">
                    Contenido de la base de datos:
                </div>
                <div id="contenido" class="bg-primary-subtle border border-primary border-5 p-3">
                    <b> Resultado de Query
                        <?php
                        $sql = "SELECT id, nombre, apellido, edad FROM persona"; // SQL query to select data from 'persona' table
                        $result = mysqli_query($conn, $sql); // Executes the query
                        if(mysqli_num_rows($result) > 0) {
                            // Output data of each row
                            while($row = mysqli_fetch_assoc($result)) {
                                echo "<br> Nombre: " . $row["nombre"] . "<br> Apellido: " . $row["apellido"] . "<br> Edad: " . $row["edad"] . "<br>";
                            }
                        } else {
                            echo "0 results";
                       }
                        ?>
                    </b>
                </div>
            </div> 
        </div>
    </div>
    <form id="myForm" method="post">
        <div class="container mt-5">
            <div class="row">
                <div class="col">
                    <div class="mb-3">
                        <label for="nombre" class="form-label">Nombre</label>
                        <input type="text" class="form-control" id="nombre" name="nombre" required>
                    </div>
                    <div class="mb-3">
                        <label for="apellido" class="form-label">Apellido</label>
                        <input type="text" class="form-control" id="apellido" name="apellido" required>
                    </div>
                    <div class="mb-3">
                        <label for="edad" class="form-label">Edad</label>
                        <input type="number" class="form-control" id="edad" name="edad" required>
                    </div>
                    <button type="button" class="btn btn-primary" id="saveChangesBtn">Elegir opción</button>
                </div>
            </div>
        </div>
    </form>
    <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
    <script>
        document.getElementById('saveChangesBtn').addEventListener('click', function() {
            Swal.fire({
                title: "Do you want to save the changes?",
                showDenyButton: true,
                showCancelButton: true,
                confirmButtonText: "Save",
                denyButtonText: `Don't save`
            }).then((result) => {
                if (result.isConfirmed) {
                    document.getElementById("myForm").submit(); // Submit the form if confirmed
                } else if (result.isDenied) {
                    // Do nothing or show a message if changes are not saved
                }
            });
        });
    </script>
</body>
</html>





