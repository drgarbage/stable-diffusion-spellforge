#!/usr/bin/env python
import asyncio
import aio_pika
import threading
import json
import aioipfs
import requests
from io import BytesIO
from multiaddr import Multiaddr


def decode_base64_to_image(encoding):
    if encoding.startswith("http://") or encoding.startswith("https://"):
        if not opts.api_enable_requests:
            raise HTTPException(status_code=500, detail="Requests not allowed")

        if opts.api_forbid_local_requests and not verify_url(encoding):
            raise HTTPException(status_code=500, detail="Request to local resource not allowed")

        headers = {'user-agent': opts.api_useragent} if opts.api_useragent else {}
        response = requests.get(encoding, timeout=30, headers=headers)
        try:
            image = Image.open(BytesIO(response.content))
            return image
        except Exception as e:
            raise HTTPException(status_code=500, detail="Invalid image url") from e

    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    try:
        image = Image.open(BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as e:
        raise HTTPException(status_code=500, detail="Invalid encoded image") from e
    

def image_buffer(image):
    buf = BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    return buf


async def upload_images(images: list):
    hashes = []
    client = aioipfs.AsyncIPFS(maddr=Multiaddr('/dns4/ai.printii.com/tcp/5001/http'))

    for image in images:
        file = image_buffer(image)
        entry = await client.core.add_bytes(file.getvalue(), to_files='/photo.png')
        filename = entry['Name']
        filehash = entry['Hash']
        hashes.append(filehash)
        print(f'File uploaded {entry}')
    
    await client.close()
    print(f'Processed images: {hashes}')
    return hashes




async def consume():
    connection = await aio_pika.connect_robust("amqp://ai.printii.com")
    
    async with connection:
        channel = await connection.channel()
        queue = await channel.declare_queue('generation', durable=False)

        print(' [x] SpellForge Generation Worker is running.')

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    print(f" [x] Processing SpellForge Generation... {message.body.decode()}")
                    from modules import scripts, shared
                    from modules.shared import opts
                    from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
                    from contextlib import closing

                    msg = json.loads(message.body.decode())
                    api = msg.get('api')
                    args = msg.get('params') 
                    reportProgress = msg.get('reportProgress')
                    reportResult = msg.get('reportResult')
                    print(f" [x] SpellForge calling {api}...")

                    if(api == 'txt2img'):
                        with closing(StableDiffusionProcessingTxt2Img(sd_model=shared.sd_model, **args)) as p:
                            p.is_api = False
                            p.outpath_grids = opts.outdir_txt2img_grids
                            p.outpath_samples = opts.outdir_txt2img_samples

                            try:
                                shared.state.begin(job="scripts_txt2img")
                                processed = process_images(p)
                                hashes = await upload_images(processed.images)
                                data = { "result": { "images": hashes } }
                                response = requests.post(reportResult, json=data)
                                print(f"Finall Result: {response}")
                            finally:
                                shared.state.end()
                                shared.total_tqdm.clear()
                                print(" [x] Done")
                    
                    if(api == 'img2img'):
                        with closing(StableDiffusionProcessingImg2Img(sd_model=shared.sd_model, **args)) as p:
                            p.init_images = # [decode_base64_to_image(x) for x in init_images]
                            p.is_api = False
                            p.outpath_grids = opts.outdir_img2img_grids
                            p.outpath_samples = opts.outdir_img2img_samples

                            try:
                                shared.state.begin(job="scripts_img2img")
                                processed = process_images(p)
                                hashes = await upload_images(processed.images)
                                data = { "result": {"images": hashes } }
                                response = requests.post(reportResult, json=data)
                                print(f"Finall Result: {response}")
                            finally:
                                shared.state.end()
                                shared.total_tqdm.clear()
                                print(" [x] Done")

                    



def run_consume_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(consume())

def preload(parser):
    parser.add_argument(
        "--spellforge-apikey",
        type=str,
        help="API key to access generation",
        default=None,
    )
    
    # Start consume function in a separate thread
    consume_thread = threading.Thread(target=run_consume_loop)
    consume_thread.start()
    print(" [x] Started the consume thread.")

    # Optionally, return the consume_thread in case you want to join it later
    return consume_thread