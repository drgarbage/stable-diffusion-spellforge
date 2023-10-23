import launch

if not launch.is_installed("aio_pika"):
    launch.run_pip("install aio_pika", "requirements for queue handling")

if not launch.is_installed("aioipfs"):
    launch.run_pip("install aioipfs", "requirements for file storage")

if not launch.is_installed("httpx"):
    launch.run_pip("install httpx", "requirements for reporting result")

if not launch.is_installed("rembg"):
    launch.run_pip("install rembg", "requirements for removing background")