-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Waktu pembuatan: 27 Agu 2024 pada 09.11
-- Versi server: 10.4.32-MariaDB
-- Versi PHP: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `anomali_detection`
--

-- --------------------------------------------------------

--
-- Struktur dari tabel `login`
--

CREATE TABLE `login` (
  `id_member` int(11) NOT NULL,
  `full_name` text NOT NULL,
  `email` varchar(255) NOT NULL,
  `password` varchar(255) NOT NULL,
  `reset_token` varchar(64) DEFAULT NULL,
  `reset_token_expiry` datetime DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data untuk tabel `login`
--

INSERT INTO `login` (`id_member`, `full_name`, `email`, `password`, `reset_token`, `reset_token_expiry`) VALUES
(128, 'elsa', 'aku2@gmail.com', '$2y$10$D/1p16uoq3G8.tdA3dJawOksrII9x3dEiJox.5UCWll4NzA6CZmbG', NULL, NULL),
(132, 'rahma', 'aku3@gmail.com', '$2y$10$K4oG3G0enhKWJprdElBK6ec6XuKcvXQQWApWjo4H6Uo60Ja0.uFxO', NULL, NULL),
(135, 'rahma', 'aku4@gmail.com', '$2y$10$rNDnWeBgK9Gs8yPacZ6HZ.5Cp9OtUVOHc0JIacM3GwZS/13OjGHjW', NULL, NULL),
(137, 'rahma', 'aku5@gmail.com', '$2y$10$RlWQ1haSQxfFArKI06usXOG0KWFnn9zxvzzc1q4SSYzoJPrTxnJue', NULL, NULL),
(138, 'elsa', 'elsarahmadani85@gmail.com', '$2y$10$xOPbFO6eYfMt/Q.dUblxLulFdlp.rebfoAocKbmkLu3SCwm7U8pia', 'e3158c41183b66d6c35f84c2922c14fe991273b35d2ab8ac7f71010bce5a89dc', '2024-08-27 15:06:04');

--
-- Indexes for dumped tables
--

--
-- Indeks untuk tabel `login`
--
ALTER TABLE `login`
  ADD PRIMARY KEY (`id_member`),
  ADD UNIQUE KEY `email` (`email`);

--
-- AUTO_INCREMENT untuk tabel yang dibuang
--

--
-- AUTO_INCREMENT untuk tabel `login`
--
ALTER TABLE `login`
  MODIFY `id_member` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=139;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
