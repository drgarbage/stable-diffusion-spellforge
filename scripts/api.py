import io
import base64
import asyncio
import threading
import json
import httpx
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
    from modules import scripts, shared
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


async def report_progress(report_url):
    try:
        progress, progressImage = get_progress()

        if progress == 0:
            return

        if progress >= 1:
            return

        progressImageHash = None
        if progressImage is not None:
            hashes = await upload_images([progressImage])
            progressImageHash = hashes[0]

        data = { "progress": progress, "progressImage": progressImageHash }

        async with httpx.AsyncClient() as client:
            response = await client.post(report_url, json=data)

    except Exception as e:
        print("[x] Error occur on reporting progress.")
        print(e)
    finally:
        await asyncio.sleep(1)


async def upload_images(images: list):
    hashes = []
    async with aioipfs.AsyncIPFS(maddr=Multiaddr('/dns4/ai.printii.com/tcp/5001/http')) as client:
        for image in images:
            file = image_buffer(image)
            entry = await client.core.add_bytes(file.getvalue(), to_files='/photo.png')
            filehash = entry['Hash']
            filename = f"{filehash}.png"
            await client.files.cp(f"/ipfs/{filehash}", f"/{filename}")
            hashes.append(filehash)
    return hashes


async def process_image(api, args):
    from contextlib import closing
    from modules import scripts, shared, ui
    from modules.shared import opts
    from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images

    script_name = args.get('script_name', None)
    script_args = args.get('script_args', [])
    alwayson_scripts = args.get('alwayson_scripts', None)
    script = None

    if api == 'txt2img':
        script_runner = scripts.scripts_txt2img
        processing_class = StableDiffusionProcessingTxt2Img
        outpath_grids = opts.outdir_txt2img_grids
        outpath_samples = opts.outdir_txt2img_samples
        job_name = "scripts_txt2img"
    elif api == 'img2img':
        script_runner = scripts.scripts_img2img
        processing_class = StableDiffusionProcessingImg2Img
        outpath_grids = opts.outdir_img2img_grids
        outpath_samples = opts.outdir_img2img_samples
        job_name = "scripts_img2img"
    else:
        raise ValueError("Invalid API")

    if not script_runner.scripts:
        script_runner.initialize_scripts(False)
        ui.create_ui()

    max_args = 1
    for script in script_runner.scripts:
        if max_args < script.args_to:
            max_args = script.args_to
    full_script_args = [None] * max_args
    full_script_args[0] = 0

    with gr.Blocks():
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
            alwayson_script_index = [script.title().lower() for script in scripts].index(alwayson_script_name.lower())
            alwayson_script = script_runner.scripts[alwayson_script_index]
            print(f"is alwayson: {alwayson_script_name}", alwayson_script.alwayson)
            if "args" in alwayson_scripts[alwayson_script_name]:
                for idx in range(0, min((alwayson_script.args_to - alwayson_script.args_from)), len(alwayson_scripts[alwayson_script_name]["args"])):
                    full_script_args[alwayson_script.args_from + idx] = alwayson_scripts[alwayson_script_name]["args"][idx]

    if api == 'img2img':
        mask = args.get('mask', None)
        if mask:
            mask = decode_base64_to_image(mask)
            args['mask'] = mask;
    
    if "sampler_index" in args and "sampler_name" not in args:
        args["sampler_name"] = args["sampler_index"]
        args.pop("sampler_index")

    for key in ['script_name', 'script_args', 'alwayson_scripts', 'send_images', 'save_images']:
        args.pop(key, None)

    processed = []

    with closing(processing_class(sd_model=shared.sd_model, **args)) as p:
        if api == 'img2img':
            p.init_images = [decode_base64_to_image(x) for x in args.get('init_images')]
        p.is_api = False
        p.outpath_grids = outpath_grids
        p.outpath_samples = outpath_samples

        try:
            shared.state.begin(job=job_name)
            p.script_args = full_script_args
            processed = script_runner.run(p, *p.script_args)
            if processed is None:
                processed = process_images(p)
        finally:
            shared.state.end()
            shared.total_tqdm.clear()

    return processed


async def on_message(message: aio_pika.IncomingMessage):
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
                return  # Skip the current iteration and move to the next message

            progress_event = threading.Event()

            async def run_report_progress():
                while not progress_event.is_set():
                    await report_progress(reportProgress)

            progress_thread = threading.Thread(target=lambda: asyncio.run(run_report_progress()))
            progress_thread.start()

            processed = await process_image(api, args)
            
            progress_event.set()

            hashes = await upload_images(processed.images)
            data = { "result": { "images": hashes } }
            async with httpx.AsyncClient() as client:
                response = await client.post(reportResult, json=data)
            print(f"[x] Finall Result: {hashes}")

        except Exception as e:
            data = { "result": { "images": [] }, "error": e }
            async with httpx.AsyncClient() as client:
                response = await client.post(reportResult, json=data)
            print("[x] Error occur on processing task.")
            print(e)


async def regist_worker():
    connection = await aio_pika.connect_robust("amqp://ai.printii.com")
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)
    queue = await channel.declare_queue('generation', durable=False)
    await queue.consume(on_message)
    print('[x] SpellForge Generation Worker is running.')
    try:
        await asyncio.Future()
    finally:
        await connection.close()


def rembg_api(app: FastAPI):

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


def regisit_services(_: gr.Blocks, app: FastAPI):
    rembg_api(app)
    asyncio.run(regist_worker())


try:
    import modules.script_callbacks as script_callbacks
    script_callbacks.on_app_started(regisit_services)
except:
    pass