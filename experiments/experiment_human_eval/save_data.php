<?php
// Ensure the data directory exists
$data_dir = "data";
if (!file_exists($data_dir)) {
    mkdir($data_dir, 0777, true);
}

// Get the raw POST data
$json_data = file_get_contents('php://input');
$data = json_decode($json_data, true);

// Ensure we have a user ID
if (!isset($data['userId'])) {
    http_response_code(400);
    echo json_encode(['success' => false, 'error' => 'Missing userId']);
    exit;
}

// Create user-specific filename
$user_filename = $data_dir . "/user_" . $data['userId'] . ".json";

// Load existing data or create new array
$existing_data = [];
if (file_exists($user_filename)) {
    $file_content = file_get_contents($user_filename);
    $existing_data = json_decode($file_content, true) ?: [];
}

// Add timestamp to the new entry
$data['timestamp'] = date('Y-m-d H:i:s');

// Append new data
$existing_data[] = $data;

// Save the updated data
file_put_contents($user_filename, json_encode($existing_data, JSON_PRETTY_PRINT));

// Return success response
echo json_encode(['success' => true, 'filename' => $user_filename]);
?>
