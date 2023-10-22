import io
import base64
import asyncio
import threading
import json
import requests
import gradio as gr
import aio_pika
import aioipfs

from io import BytesIO
from PIL import PngImagePlugin, Image
from fastapi import FastAPI
from rembg import remove
from multiaddr import Multiaddr


def decode_base64_to_image(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    try:
        image = Image.open(BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as e:
        return None
    
    
def image_buffer(image):
    from modules.shared import opts
    use_metadata = False
    metadata = PngImagePlugin.PngInfo()
    for key, value in image.info.items():
        if isinstance(key, str) and isinstance(value, str):
            metadata.add_text(key, value)
            use_metadata = True
    
    buf = BytesIO()            
    image.save(
        buf, 
        format="PNG", 
        pnginfo=(metadata if use_metadata else None), 
        quality=opts.jpeg_quality)

    buf.seek(0)
    return buf


def get_progress():
    if shared.state.job_count == 0:
        return 0, None

    # avoid dividing zero
    progress = 0.01
    progressImage = None

    if shared.state.job_count > 0:
        progress += shared.state.job_no / shared.state.job_count
    if shared.state.sampling_steps > 0:
        progress += 1 / shared.state.job_count * shared.state.sampling_step / shared.state.sampling_steps

    progress = min(progress, 1)

    shared.state.set_current_image()
    if shared.state.current_image:
        progressImage = shared.state.current_image

    return progress, progressImage


async def upload_images(images: list):
    hashes = []
    async with aioipfs.AsyncIPFS(maddr=Multiaddr('/dns4/ai.printii.com/tcp/5001/http')) as client:
        for image in images:
            file = image_buffer(image)
            entry = await client.core.add_bytes(file.getvalue(), to_files='/photo.png')
            filename = entry['Name']
            filehash = entry['Hash']
            hashes.append(filehash)
            print(f'File uploaded {entry}')
    
    print(f'Processed images: {hashes}')
    return hashes



def process_txt2img(args):
    from modules import scripts, shared
    from modules.shared import opts
    from modules.processing import StableDiffusionProcessingTxt2Img, process_images
    from contextlib import closing
    script_name = args.get('script_name', None)
    script_args = args.get('script_args', [])
    alwayson_scripts = args.get('alwayson_scripts', None)
    script = None

    script_runner = scripts.scripts_txt2img
    if not script_runner.scripts:
        script_runner.initialize_scripts(False)

    # init script args
    max_args = 1
    for script in script_runner.scripts:
        if script.args_to is not None and max_args < script.args_to:
            max_args = script.args_to
            
    full_script_args = [None]*max_args
    full_script_args[0] = 0

    # get default values
    with gr.Blocks(): # will throw errors calling ui function without this
        for script in script_runner.scripts:
            if script.ui(script.is_img2img):
                ui_default_values = []
                for elem in script.ui(script.is_img2img):
                    ui_default_values.append(elem.value)
                full_script_args[script.args_from:script.args_to] = ui_default_values
                

    if script_name:
        script_index = [script.title().lower() for script in script_runner.selectable_scripts].index(script_name.lower())
        script = script_runner.selectable_scripts[script_index]
        full_script_args[script.args_from:script.args_to] = script_args
        full_script_args[0] = script_index + 1

    if alwayson_scripts:
        for alwayson_script_name in alwayson_scripts.keys():
            alwayson_script = script_runner.scripts[[script.title().lower() for script in script_runner.scripts].index(alwayson_script_name.lower())]
            if "args" in alwayson_scripts[alwayson_script_name]:
                for idx in range(0, min((alwayson_script.args_to - alwayson_script.args_from), len(alwayson_scripts[alwayson_script_name]['args']))):
                    full_script_args[alwayson_script.args_from + idx] = alwayson_scripts[alwayson_script_name]['args'][idx]

    args.pop('script_name', None)
    args.pop('script_args', None)
    args.pop('alwayson_scripts', None)
    args.pop('send_images', True)
    args.pop('save_images', None)

    processed = []

    with closing(StableDiffusionProcessingTxt2Img(sd_model=shared.sd_model, **args)) as p:
        p.is_api = False
        p.outpath_grids = opts.outdir_txt2img_grids
        p.outpath_samples = opts.outdir_txt2img_samples

        try:
            shared.state.begin(job="scripts_txt2img")
            if script_name is not None:
                p.script_args = full_script_args
                processed = scripts.scripts_txt2img.run(p, *p.script_args)
            else:
                p.script_args = tuple(full_script_args)
                processed = process_images(p)
        finally:
            shared.state.end()
            shared.total_tqdm.clear()

    return processed


def process_img2img(args):
    from modules import scripts, shared
    from modules.shared import opts
    from modules.processing import StableDiffusionProcessingImg2Img, process_images
    from contextlib import closing
    script_name = args.get('script_name', None)
    script_args = args.get('script_args', [])
    alwayson_scripts = args.get('alwayson_scripts', None)
    script = None

    script_runner = scripts.scripts_txt2img
    if not script_runner.scripts:
        script_runner.initialize_scripts(False)

    # init script args
    max_args = 1
    for script in script_runner.scripts:
        if script.args_to is not None and max_args < script.args_to:
            max_args = script.args_to
            
    full_script_args = [None]*max_args
    full_script_args[0] = 0

    # get default values
    with gr.Blocks(): # will throw errors calling ui function without this
        for script in script_runner.scripts:
            if script.ui(script.is_img2img):
                ui_default_values = []
                for elem in script.ui(script.is_img2img):
                    ui_default_values.append(elem.value)
                full_script_args[script.args_from:script.args_to] = ui_default_values
                

    if script_name:
        script_index = [script.title().lower() for script in script_runner.selectable_scripts].index(script_name.lower())
        script = script_runner.selectable_scripts[script_index]
        full_script_args[script.args_from:script.args_to] = script_args
        full_script_args[0] = script_index + 1

    if alwayson_scripts:
        for alwayson_script_name in alwayson_scripts.keys():
            alwayson_script = script_runner.scripts[[script.title().lower() for script in script_runner.scripts].index(alwayson_script_name.lower())]
            if "args" in alwayson_scripts[alwayson_script_name]:
                for idx in range(0, min((alwayson_script.args_to - alwayson_script.args_from), len(alwayson_scripts[alwayson_script_name]['args']))):
                    full_script_args[alwayson_script.args_from + idx] = alwayson_scripts[alwayson_script_name]['args'][idx]

    args.pop('script_name', None)
    args.pop('script_args', None)
    args.pop('alwayson_scripts', None)
    args.pop('send_images', True)
    args.pop('save_images', None)

    processed = []

    with closing(StableDiffusionProcessingImg2Img(sd_model=shared.sd_model, **args)) as p:
        p.init_images = [decode_base64_to_image(x) for x in args.get('init_images')]
        p.is_api = False
        p.outpath_grids = opts.outdir_img2img_grids
        p.outpath_samples = opts.outdir_img2img_samples

        try:
            shared.state.begin(job="scripts_img2img")
            if script_name is not None:
                p.script_args = full_script_args
                processed = scripts.scripts_img2img.run(p, *p.script_args)
            else:
                p.script_args = tuple(full_script_args)
                processed = process_images(p)
        finally:
            shared.state.end()
            shared.total_tqdm.clear()

    return processed


async def report_progress(report_url):
    while True:
        progress, progressImage = get_progress()

        progressImageHash = None
        if progressImage is not None:
            hashes = await upload_images([progressImage])
            progressImageHash = hashes[0]

        data = { "progress": progress, "progressImage": progressImageHash }

        try:
            response = requests.post(report_url, json=data)
            print(f"[x] Progress: {progress}% - {response}")
        except Exception as e:
            print("[x] Error occur on reporting progress.")
            print(e)
        await asyncio.sleep(1)


async def consume():
    connection = await aio_pika.connect_robust("amqp://ai.printii.com")
    
    async with connection:
        channel = await connection.channel()
        queue = await channel.declare_queue('generation', durable=False)

        print('[x] SpellForge Generation Worker is running.')

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    try:
                        print(f"[x] Accepted Generation Request...")

                        msg = json.loads(message.body.decode())
                        api = msg.get('api')
                        args = msg.get('params') 
                        reportProgress = msg.get('reportProgress')
                        reportResult = msg.get('reportResult')

                        print(f"[x] Processing SpellForge {api}...")

                        if not api:
                            print("[x] API is None. Skipping this message.")
                            continue  # Skip the current iteration and move to the next message

                        progress_task = asyncio.create_task(report_progress(reportProgress))
                        processed = []

                        if api == 'txt2img':
                            processed = process_txt2img(args)

                        if api == 'img2img':
                            processed = process_img2img(args)

                        progress_task.cancel()

                        if processed.images:
                            hashes = await upload_images(processed.images)
                        else:
                            hashes = []
                        print(f"hashes {hashes}")
                        data = { "result": { "images": hashes } }
                        response = requests.post(reportResult, json=data)
                        print(f"[x] Finall Result: {response}")
                    except Exception as e:
                        print("[x] Error occur on processing task.")
                        print(e)
                    

def run_consume_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(consume())


def regist_worker():
    # Start consume function in a separate thread
    consume_thread = threading.Thread(target=run_consume_loop)
    consume_thread.start()

    # Optionally, return the consume_thread in case you want to join it later
    return consume_thread


def rembg_api(_: gr.Blocks, app: FastAPI):

    @app.get("/sdapi/v1/rembg/version")
    async def version():
        return {"version": "1.0"}

    @app.post("/sdapi/v1/rembg/remove_background")
    async def remove_background(image_data: dict):
        try:
            # Read upload image
            base64_data = image_data.get("image")
            if not base64_data:
                raise ValueError("No image data provided")
            
            # Convert base64 to image
            content = base64.b64decode(base64_data)
            input_image = Image.open(io.BytesIO(content))

            # Remove background
            output_image = remove(input_image)

            # Convert to byte and return
            output_byte_array = io.BytesIO()
            output_image.save(output_byte_array, format="PNG")

            # Convert to base64
            output_base64 = base64.b64encode(output_byte_array.getvalue()).decode("utf-8")

            return {"success": True, "image": output_base64}
        except Exception as e:
            return {"success": False, "error": str(e)}

    regist_worker()


try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(rembg_api)
except:
    pass