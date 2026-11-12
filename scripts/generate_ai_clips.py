#!/usr/bin/env python3
"""Generate extra AI clips with Wan2.2-TI2V-5B for the next labeling phase.

Matches the existing ``dataset/ai`` format: 1280x720 FFV1 yuv420p, 24 fps,
121 frames, plus a JSON sidecar and a manifest line per clip. Resumable:
clips whose .mkv already exists are skipped.

``--res 1080p`` renders the same prompts natively at 1920x1088 (Wan latent grid is a
multiple of 32) and centre-crops to 1920x1080 (no resampling). Ids get a ``_1080``
suffix so ``clip_family`` groups each clip with its 720p sibling (same split).

Run with the Wan env:
  /workspace/dataset-src/wan-env/bin/python scripts/generate_ai_clips.py
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# 1080p VAE decode needs large contiguous blocks; avoid "reserved but unallocated" fragmentation OOMs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

WAN_ROOT = Path("/workspace/dataset-src/Wan2.2")
CKPT = Path("/workspace/dataset-src/weights/Wan2.2-TI2V-5B")
sys.path.insert(0, str(WAN_ROOT))

LICENCE = "Apache-2.0 model, outputs usable for training"

HARD = [
    "Blizzard in a pine forest, thousands of snowflakes streaking past the camera, gusting wind, handheld",
    "Sandstorm sweeping across desert dunes, dense swirling grains and dust filling the frame, low camera",
    "Close-up of a powerful waterfall crashing into a pool, white spray and churning foam everywhere",
    "Ocean waves smashing against jagged black rocks, explosive white spray, fast handheld camera",
    "Night fireworks finale, hundreds of bursting sparkling shells filling the sky, smoke and glitter",
    "Close-up of glitter swirling in a shaken snow globe, thousands of sparkling flakes tumbling",
    "Thick white smoke billowing and swirling from a smoke grenade in a forest, turbulent wisps",
    "Welding sparks showering from a grinder in a dark workshop, thousands of bright streaks, close-up",
    "A dense swarm of bees around a hive, thousands of wings and bodies moving, extreme close-up",
    "Heavy rain streaming down a window at night with blurred city lights, countless running droplets",
    "Fast handheld run through a crowded night market, neon signs, lanterns and people rushing past",
    "Underwater close-up of thousands of rising bubbles in turbulent water, sunlight flickering",
    "Autumn leaves blizzard, thousands of red and yellow leaves blowing across the frame in strong wind",
    "Packed football stadium crowd waving flags and jumping, seen from close above, confetti falling",
    "Fast whip pans across a busy city intersection at rush hour, cars, buses and pedestrians",
    "Camera speeding along a chain-link fence with a sunlit field behind it, fine wire mesh flickering",
    "A herd of zebras galloping across dusty savanna, dense black and white stripes in motion, tracking shot",
    "Fast drone flight low over a dense coniferous forest canopy, thousands of sharp treetops rushing below",
    "Close-up of a fast flowing mountain stream over pebbles, sparkling ripples and caustics everywhere",
    "Hundreds of starlings murmuration swirling over a reed bed at dusk, constantly changing shapes",
    "Close-up of popcorn violently popping in a pot, kernels bursting and flying everywhere",
    "Torrential rain on a lake surface, thousands of splashes and ripples, close-up low angle",
    "Fast handheld walk through a dense bamboo forest with flickering sunlight, leaves shaking",
    "Burning forest at night, towering flames, embers and smoke swirling, wide handheld shot",
    "Close-up of boiling sea foam and kelp in a rough tide pool, rapid chaotic motion",
    "Thousands of tiny colourful beads pouring and bouncing onto a glass table, macro close-up",
    "Snow avalanche thundering down a mountain slope, billowing powder cloud filling the frame",
    "A crowded marathon start, thousands of runners surging forward, seen from a low moving camera",
    "Close-up of wheat ears and dust in a combine harvester's spray, chaotic motion, sunset light",
    "Shaky first-person mountain bike descent through a rocky forest trail, branches whipping past",
    "Rain-soaked neon city street at night, reflections rippling in puddles, cars splashing past",
    "Close-up of a tree full of cherry blossoms in a gale, thousands of petals swirling away",
    "Dense flock of pigeons taking off from a cobblestone square, hundreds of flapping wings",
    "A flooded river in spate, brown churning water with debris rushing past, fast handheld shot",
    "Fine sand pouring from a hand in strong backlight, thousands of glittering grains, macro shot",
    "Dancers in sequinned costumes spinning under flashing stage lights, glitter and motion blur",
    "Close-up of a wind-whipped field of lavender, thousands of purple stems shaking, bees flying",
    "Fast tracking shot along a busy rail yard, tracks, wires and gravel flickering past",
    "Hailstorm on a car windshield, thousands of ice pellets bouncing, wipers moving fast",
    "Swirling cloud of autumn dust and seeds in a sunbeam inside a barn, dense flickering particles",
]

MEDIUM = [
    "A man jogging along a riverside path in the morning, smooth side tracking shot",
    "A sailboat gliding across a calm bay, gentle waves, slow aerial orbit",
    "Children playing in a playground, medium shot, gentle handheld camera",
    "A slow walk through a library aisle with tall bookshelves, warm light",
    "Traffic on a highway at dusk seen from an overpass, steady wide shot",
    "A street musician playing guitar on a city corner, a few people passing by",
    "Horses grazing in a meadow, light breeze, slow pan",
    "A barista pouring latte art in a cafe, close-up, steady shot",
    "A slow drone shot over a small harbour with fishing boats",
    "Rain falling gently on a garden path, medium shot, steady camera",
    "A tram moving through a historic city street, static shot",
    "Snow falling softly on a quiet village street at night, slow push-in",
    "A family walking a dog on a beach at sunset, smooth follow shot",
    "Ducks swimming across a pond with reeds, gentle ripples, static shot",
    "A skateboarder cruising along a boardwalk, smooth tracking shot",
]

EASY = [
    "A cup of coffee on a desk next to a laptop, static shot, soft daylight",
    "A potted succulent on a white windowsill, static close-up, no movement",
    "A calm foggy field at sunrise, static wide shot, almost no motion",
    "An empty conference room with a whiteboard, static shot, even lighting",
    "A stack of books on a wooden table, slow gentle push-in, warm light",
    "A lighthouse on a cliff under a clear sky, static wide shot",
    "A glass of water on a marble counter, static macro shot",
    "A parked bicycle against a brick wall, static shot, afternoon light",
    "A desert landscape under a clear blue sky, very slow pan",
    "A wall clock in a quiet room, static shot, only the second hand moves",
]


# Animated-movie hard clips. What makes a clip expensive at 0.05 bpp is high-frequency detail
# that moves: fur and hair strands, fabric weave, foliage, particles, rain, sparkles, caustics,
# render grain, plus a fast or shaking camera and many small moving figures. Flat cel shading,
# big smooth gradients and slow cameras are cheap, so every prompt below asks for textured,
# "rendered" detail and keeps the camera moving. Style words that Wan 2.2 follows well are put
# first; each prompt names subject, texture, motion, camera and light.
ANIM_HARD = [
    # 3D feature-film renders: fur, hair, cloth, foliage, particles
    "Pixar-style 3D animated film, highly detailed render, a red fox with dense wind-blown fur sprinting through a dry autumn forest, thousands of sharp leaves whirling around it, fast low tracking camera, dappled sunlight, subtle film grain",
    "Pixar-style 3D animated film, a shaggy blue monster with long fur shaking rain off its coat in a storm, every strand and droplet visible, water spray, handheld camera close-up, dramatic rim light",
    "DreamWorks-style 3D animation, a dragon beating its wings over a burning village, ember storms and thick textured smoke, scales glinting, fast orbiting camera, night",
    "3D animated feature, extremely detailed render, a tiny mouse running across a cluttered wooden kitchen floor, flour dust, scattered grains and crumbs, fast camera at floor level, warm kitchen light",
    "3D animated feature, a flock of hundreds of colourful parrots exploding out of a dense jungle canopy, individual feathers and leaves everywhere, camera flying with the flock, strong backlight",
    "3D animated feature, a stampede of woolly bison across a dusty plain, thick textured fur, dust clouds, grass and pebbles kicked up, shaking aerial tracking shot, golden hour",
    "3D animated feature, an underwater reef with thousands of glittering small fish swirling in a bait ball, bubbles, light caustics rippling over coral, fast gliding camera",
    "3D animated feature, a wizard's sandstorm spell in a desert market, swirling sand grains filling the air, fluttering striped awnings and carpets, fast handheld camera, harsh sunlight",
    "3D animated feature, a robot racing a go-kart through a junkyard of rusty textured scrap, sparks, mud splatter and flying debris, fast chase camera, overcast light",
    "3D animated feature, a squirrel leaping between snow-covered pine branches, snow powder bursting from every branch, thousands of needles, fast whip-pan camera, cold blue light",
    "3D animated feature, a festival in a medieval town with hundreds of small cheering characters in patterned clothes, confetti and streamers filling the air, sweeping crane shot, torchlight",
    "3D animated feature, close-up of a fluffy kitten pouncing through tall meadow grass full of dandelion seeds and insects, fur and grass blades sharp, fast handheld camera, backlit",
    # anime with textured, detailed rendering (not flat)
    "detailed anime film style with painted textures, a samurai duel inside a storm of cherry blossom petals, rain and sparks from clashing blades, speed lines, dynamic rotating camera",
    "detailed anime film style, a giant robot battle in a crumbling city, concrete debris, dust, glass shards and smoke everywhere, violent shaking camera, dusk",
    "detailed anime film style, a girl running through a night summer festival crowd, hundreds of lanterns, fireworks bursting overhead, yukata patterns and crowd faces, fast tracking camera",
    "detailed anime film style, a bullet train racing through a rainstorm, rain streaks and splashes on the window, wet rice fields flashing past, camera riding alongside",
    "detailed anime film style, a magical battle with thousands of glowing particles, energy beams and shattering crystals, intense flicker, spinning camera",
    "detailed anime film style, a cyclist speeding downhill through a sunflower field, petals and insects flying, strong wind in the hair and clothes, low fast tracking camera, noon sun",
    "detailed anime film style, an airship chase through thunderclouds, rain, lightning flashes, spinning propellers and smoke trails, fast camera, dramatic lighting",
    "detailed anime film style, a snowstorm on a mountain pass, two travellers in fur coats struggling forward, snow whipping across the frame, handheld camera, grey light",
    # stop-motion and textured handmade looks (fabric, clay, paper grain)
    "stop-motion claymation film, fingerprint-textured clay characters, a market scene with dozens of clay figures bustling, flags and bunting flapping, puddles splashing in rain, dolly shot",
    "stop-motion felt puppet film, a snowball fight between a dozen fuzzy felt animals, hundreds of cotton snowballs flying, visible fabric fibres, fast camera, bright winter light",
    "stop-motion paper cut-out film, an autumn street with hundreds of falling paper leaves of different textures, gusting wind, panning camera, late afternoon light",
    "stop-motion claymation film, a clay volcano erupting with chunks of glowing clay lava, cotton smoke and glitter sparks, camera shaking, night",
    # painterly and mixed-media looks (brush texture everywhere)
    "oil-painting animated film, every frame visibly brush-textured, a storm at sea with a whale breaching, spray, rain and foam, sweeping camera, dark sky",
    "watercolour animated film with paper grain, a wildflower meadow in strong wind with swarming butterflies, camera gliding low through the stems, soft sunlight",
    "cel-shaded 3D anime style with detailed textures, a motorcycle race through a desert canyon, dust trails, sparks and speed lines, fast tracking camera, sunset",
    "mixed-media animated film, hand-drawn characters over a grainy photographic city at night, rain, neon reflections and traffic, fast handheld camera",
    "3D animated feature, a marching toy-soldier army crossing a thick woollen rug, hundreds of tiny figures, dust motes in a sunbeam, low dolly shot",
    "3D animated feature, a hummingbird hovering among wet jungle flowers after rain, water droplets flying from beating wings, rich textured leaves, macro camera drifting fast",
]

# Human-like animated films: realistic CG humans are the hardest animated content for an encoder
# (hair strands, skin texture, cloth folds, crowds, sweat and rain on faces), and the hidden set is
# described as AI clips with genuine full-resolution detail.
ANIM_HUMAN = [
    "photorealistic CGI animated film in the style of a modern game cinematic, a young woman with long wind-blown hair running through a rain-soaked neon street at night, water droplets on her face, wet reflective asphalt, handheld camera following close",
    "Pixar-style 3D animated film, a close-up of a laughing grandmother with detailed skin, freckles and curly grey hair, knitted wool cardigan, sunlight through a window with floating dust, slow handheld camera",
    "DreamWorks-style 3D animated film, a group of teenagers dancing in a crowded gym under spinning disco lights, sequinned clothes, flying hair, confetti, fast camera moving through the crowd",
    "photorealistic CGI animated film, a sprinter in a textured running vest exploding out of the blocks, sweat and water droplets flying, crowd in the stands, fast tracking camera, stadium lights",
    "3D animated feature, a market street full of hundreds of human characters in patterned fabrics bargaining and walking, flags fluttering, pigeons taking off, sweeping crane shot, midday sun",
    "photorealistic CGI animated film, a knight in battered chainmail and fur cloak charging through a rain-lashed battlefield, mud splashing, torn banners whipping, shaking handheld camera, grey light",
    "Pixar-style 3D animated film, two children in rain jackets jumping in puddles during a downpour, water splashing everywhere, hair plastered to their faces, fast low camera",
    "3D animated feature, extremely detailed render, a girl brushing a horse in a dusty barn, hair of both catching the sunlight, straw and dust particles swirling, slow orbiting camera",
    "photorealistic CGI animated film, a street musician with a grey beard playing violin in falling snow, every snowflake and beard hair visible, bokeh city lights, slow push-in",
    "Pixar-style 3D animated film, a chef in a chaotic restaurant kitchen flipping vegetables in a flaming pan, steam, sparks and flying herbs, whip-pan handheld camera",
    "photorealistic CGI animated film, a surfer with dripping hair riding inside a barrelling wave, spray and foam everywhere, camera in the water tracking alongside, bright sun",
    "3D animated feature, a classroom of thirty children throwing paper planes and confetti, patterned shirts, fast camera weaving between desks, afternoon light",
    "photorealistic CGI animated film, a boxer being hit in slow motion then real time, sweat spraying, ropes vibrating, crowd and camera flashes in the background, dynamic camera",
    "Pixar-style 3D animated film, a family picnic in a windy meadow, blankets and hair blowing, hundreds of grass blades and dandelion seeds, dog running through, tracking camera",
    "photorealistic CGI animated film, a woman in a flowing patterned silk dress spinning on a rooftop at sunset, fabric and hair swirling, city lights flickering on, orbiting camera",
    "3D animated feature, a snowball battle between a dozen kids in woollen hats and scarves, snow bursting on impact, breath fog, handheld camera, overcast light",
    "photorealistic CGI animated film, a firefighter with a soot-covered face running out of a burning building, embers and ash swirling, water spray from hoses, shaking camera",
    "Pixar-style 3D animated film, a marathon crowd of hundreds of runners in colourful kit crossing a bridge in the rain, splashes and wet reflections, low moving camera",
    "photorealistic CGI animated film, a close-up of an old fisherman's weathered face and knitted hat in sea spray on a rocking boat, gulls behind, handheld camera",
    "3D animated feature, a wedding party dancing outdoors at night under strings of lights, lace and sequins, flying petals, long hair swinging, fast camera through the dancers",
]

# A few calmer animated clips so the label set covers the whole difficulty range for this style.
ANIM_MEDIUM = [
    "Pixar-style 3D animated film, a boy reading in a treehouse at dusk, leaves gently moving outside, warm lamp light, slow push-in",
    "3D animated feature, a quiet harbour village with a few boats bobbing, laundry swaying in a light breeze, slow aerial drift",
    "detailed anime film style, a girl walking a dog along a riverside path in the morning, gentle side tracking shot",
    "stop-motion claymation film, a clay baker decorating a cake in a tidy kitchen, static camera, soft light",
    "3D animated feature, a cat sleeping on a sunny windowsill, curtains barely moving, static camera",
    "watercolour animated film, a slow sunrise over calm hills with a single drifting balloon, slow pan",
    "Pixar-style 3D animated film, a robot watering plants in a greenhouse, slow dolly, soft diffused light",
    "detailed anime film style, a quiet classroom after school, dust in sunbeams, a student packing a bag, static camera",
]

# Native high-frame-rate AI clips: Wan samples 121 frames at its 24 fps training rate; slow-motion
# prompts cut the per-frame motion ~2-3x, i.e. the frame-to-frame statistics of real 50-60 fps footage
# at normal speed (not interpolated). Stored at the listed fps. Entries: (fps, style, prompt).
HFR_WAN_HARD = [
    (60, "live", "Slow motion close-up of a hummingbird hovering beside a red flower, wings beating, tiny water droplets drifting through the air, shallow depth of field, high-speed camera footage"),
    (60, "live", "Slow motion of a surfer carving through a large breaking wave, spray and foam flying toward the lens, bright sunlight, tracking shot, high-speed camera footage"),
    (50, "live", "Slow motion of a crowd at a night concert, confetti and streamers falling through colourful stage lights and haze, hands raised, handheld camera"),
    (60, "animated", "3D animated feature film in slow motion, a fox leaping across a rushing forest stream, water splashing, leaves swirling, tracking camera"),
]
HFR_WAN_MEDIUM = [
    (60, "live", "Slow motion of a cyclist riding along a coastal road at sunset, smooth side tracking shot, waves rolling in the background, high-speed camera footage"),
    (50, "live", "Slow motion of a chef tossing vegetables in a flaming wok, sparks and steam rising, warm kitchen light, medium close-up"),
    (60, "live", "Slow motion of a basketball player dribbling and leaping for a dunk in an indoor gym, side tracking camera, high-speed camera footage"),
    (60, "animated", "anime film style in slow motion, a girl's hair and scarf blowing in the wind on a rooftop at dusk, city lights below, slow orbit camera"),
]
HFR_WAN_EASY = [
    (60, "live", "Slow motion of coffee being poured into a white cup, cream swirling, soft morning window light, static camera, high-speed camera footage"),
    (50, "live", "Slow motion of a sailboat gliding across a calm lake at dawn, gentle ripples, mist over the water, slow pan"),
]

# set gen3 (2026-11-10, user: "AI native clips for 1080"): native 1920x1088 only; the 720p twin is a Lanczos downscale
# of the same frames. Content the earlier sets lack: real-looking people (portrait, interview, walking), product,
# food, vehicles, sci-fi / cyberpunk, macro VFX (ink), wildlife, timelapse, interiors; more medium/easy tiers; mixed
# rates 24/25/30 fps (121 frames) + 4 slow-motion prompts stored at 50/60 fps.
GEN3_HARD = [
    (24, "live", "Cinematic drone shot racing low over a jagged coastline at golden hour, waves breaking white over dark rocks, sea spray in the air, fast forward motion"),
    (30, "live", "A rally car drifting through a muddy forest corner, mud and gravel spraying toward the camera, low tracking shot, overcast daylight"),
    (24, "animated", "Sci-fi 3D animated film, a spaceship weaving through a dense asteroid field, sparks and tumbling debris, glowing blue engines, fast chase camera"),
    (24, "live", "Flying camera over a cyberpunk city at night, hundreds of neon signs and holograms, rain falling, traffic streaking far below, fast flythrough"),
    (25, "live", "Macro close-up of coloured ink clouds blooming and swirling in clear water, turbulent plumes of magenta, cyan and gold, very high detail"),
    (24, "live", "A herd of wildebeest crossing a river, water splashing, dust and dense bodies pushing forward, telephoto lens, handheld camera"),
    (30, "animated", "detailed anime film style, a crowded summer festival street with fireworks bursting overhead, paper lanterns, confetti and running people, fast panning camera"),
    (60, "live", "Slow motion close-up of a water balloon bursting in bright sunlight, water splashing outward in all directions, high-speed camera footage"),
]
GEN3_MEDIUM = [
    (24, "live", "Portrait of a young woman with freckles smiling at the camera in a sunlit cafe, shallow depth of field, slight handheld movement"),
    (30, "live", "Close-up of a chef searing a steak in a cast-iron pan, butter sizzling and steam rising, warm kitchen light, slow push-in"),
    (24, "live", "A luxury perfume bottle slowly rotating on a black reflective surface, soft light streaks gliding across the glass, product commercial"),
    (25, "live", "A red sports car driving along a winding mountain road through pine forest, aerial follow shot, late afternoon sun"),
    (24, "animated", "3D animated film, a small round robot exploring a colourful alien garden with glowing plants, gentle tracking camera, soft light"),
    (30, "live", "An elderly man talking and gesturing during an interview, seated in a cosy living room, static medium shot, natural window light"),
    (24, "live", "Translucent jellyfish drifting in a dark aquarium tank, glowing pulsing bodies and trailing tentacles, slow camera drift"),
    (24, "live", "A woman in a flowing white dress walking through a wheat field at sunset, slow side tracking shot, wind moving the grain"),
    (25, "live", "Timelapse of clouds rolling over mountain peaks, cloud shadows sweeping across green valleys, wide shot"),
    (60, "live", "Slow motion of a dancer spinning in a bright studio, skirt flaring out, hair flowing, soft light, high-speed camera footage"),
    (50, "live", "Slow motion of a golden retriever running across a grass field toward the camera, ears flapping, sunny day, tracking shot"),
    (60, "animated", "3D animated film in slow motion, a paper airplane gliding over a sunlit city park, trees and people below, gentle follow camera"),
]
GEN3_EASY = [
    (24, "live", "A modern minimalist living room interior, slow dolly forward, soft morning light through large windows"),
    (30, "live", "A smartphone standing on a wooden desk with its screen glowing, static product shot, soft studio light"),
    (24, "animated", "detailed anime film style, a quiet rural train station at dusk, a light breeze moving the grass, static wide shot"),
    (25, "live", "A golden retriever lying on a wooden porch, ears twitching, breathing slowly, static shot, afternoon light"),
]

PROMPT_SETS = {
    # set name -> list of (tier, prompts, seed0, id prefix)
    "base": [("hard", HARD, 4001, "aix"), ("medium", MEDIUM, 5001, "aix"), ("easy", EASY, 6001, "aix")],
    "anim": [("hard", ANIM_HARD, 7001, "anim"), ("medium", ANIM_MEDIUM, 7501, "anim")],
    "animh": [("hard", ANIM_HUMAN, 8001, "animh")],
    "hfrwan": [("hard", HFR_WAN_HARD, 9001, "hfrwan"), ("medium", HFR_WAN_MEDIUM, 9101, "hfrwan"),
               ("easy", HFR_WAN_EASY, 9201, "hfrwan")],
    "gen3": [("hard", GEN3_HARD, 10001, "aig3"), ("medium", GEN3_MEDIUM, 10101, "aig3"),
             ("easy", GEN3_EASY, 10201, "aig3")],
}
# 720p twin of a 1080p generation: same 1920x1088 frames, centre crop to 1920x1080, Lanczos downscale
DERIVED_720P_VF = "crop=1920:1080,scale=1280:720:flags=lanczos"


# resolution -> generation size (W, H), output size, ffmpeg geometry filter, id suffix, seed offset
RESOLUTIONS = {
    "720p": {"gen": (1280, 704), "out": (1280, 720), "vf": "scale=1280:720:flags=lanczos", "suffix": "", "seed_off": 0},
    "1080p": {"gen": (1920, 1088), "out": (1920, 1080), "vf": "crop=1920:1080", "suffix": "_1080", "seed_off": 100000},
}


def build_jobs(prompt_set: str = "base", res: str = "720p") -> list[dict]:
    jobs = []
    r = RESOLUTIONS[res]
    for tier, prompts, seed0, prefix in PROMPT_SETS[prompt_set]:
        for i, p in enumerate(prompts, 1):
            fps, style, prompt = p if isinstance(p, tuple) else (24, None, p)
            jobs.append({"id": f"{prefix}_{tier}_{i:03d}{r['suffix']}", "tier": tier, "prompt": prompt,
                         "seed": seed0 + i + r["seed_off"], "fps": fps, "style": style})
    order = {"hard": 0, "medium": 1, "easy": 2}
    return sorted(jobs, key=lambda j: (order[j["tier"]], j["id"]))


def write_ffv1(video, out: Path, vf: str = "scale=1280:720:flags=lanczos", fps: int = 24) -> None:
    """video: torch tensor (C, T, H, W) in [-1, 1]."""
    import torch

    frames = ((video.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    frames = frames.permute(1, 2, 3, 0).contiguous().cpu().numpy()  # T,H,W,C
    t, h, w, _ = frames.shape
    tmp = out.with_suffix(".tmp.mkv")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-vf", f"{vf},format=yuv420p",
        "-c:v", "ffv1", "-level", "3", "-threads", "4", str(tmp),
    ]
    subprocess.run(cmd, input=frames.tobytes(), check=True)
    tmp.rename(out)


def tiled_vae_decode(orig_decode, n_tiles: int, overlap: int = 16, max_tiles: int = 4):
    """Wrap ``WanVAE.decode`` to decode the latent in ``n_tiles`` overlapping vertical strips.

    Each strip runs the full causal temporal decode on its own (so peak VRAM scales with strip width);
    strips are moved to CPU and blended with complementary linear ramps over ``overlap`` latent columns
    (16 latent = 256 px). On CUDA OOM the same latent is retried with one more strip (diffusion work kept).
    """
    import torch

    def decode(zs):
        outs = []
        for z in zs:  # (C, T, H, W) latent
            w_lat = z.shape[-1]
            n = n_tiles
            while True:
                try:
                    outs.append(_decode_strips(z, n, w_lat))
                    break
                except torch.OutOfMemoryError:
                    gc.collect()
                    torch.cuda.empty_cache()
                    if n >= max_tiles:
                        raise
                    n += 1
                    print(f"  VAE decode OOM -> retry with {n} strips", flush=True)
        return outs

    def _decode_strips(z, n, w_lat):
        if n <= 1:
            return orig_decode([z])[0]
        tile = -(-(w_lat + overlap * (n - 1)) // n)
        starts = [round(i * (w_lat - tile) / (n - 1)) for i in range(n)]
        acc = wsum = None
        for st in starts:
            piece = orig_decode([z[..., st:st + tile]])[0].float().cpu()  # (3, T, H*16, tile*16)
            scale = piece.shape[-1] // tile
            ramp = torch.ones(piece.shape[-1])
            ov = overlap * scale
            if st > 0:
                ramp[:ov] = torch.linspace(0.0, 1.0, ov + 2)[1:-1]
            if st + tile < w_lat:
                ramp[-ov:] = torch.linspace(1.0, 0.0, ov + 2)[1:-1]
            if acc is None:
                acc = torch.zeros(*piece.shape[:-1], w_lat * scale)
                wsum = torch.zeros(w_lat * scale)
            acc[..., st * scale:(st + tile) * scale] += piece * ramp
            wsum[st * scale:(st + tile) * scale] += ramp
            del piece
            torch.cuda.empty_cache()
        return acc / wsum

    return decode


def wait_for_gpu(min_free_gib: float, max_wait_s: float = 1800.0) -> None:
    """Block (up to ``max_wait_s``) until ``min_free_gib`` of VRAM is free (other GPU users)."""
    import torch

    t0, said = time.time(), False
    while True:
        gc.collect()
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info()[0] / 2**30
        if free >= min_free_gib or time.time() - t0 > max_wait_s:
            return
        if not said:
            print(f"  waiting for VRAM: {free:.1f} GiB free < {min_free_gib} GiB", flush=True)
            said = True
        time.sleep(30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/workspace/dataset/ai_gen2")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--tiers", default="hard,medium,easy")
    ap.add_argument("--set", default="base", choices=sorted(PROMPT_SETS))
    ap.add_argument("--offload", action="store_true", help="Also keep T5 on CPU (less VRAM, slower)")
    ap.add_argument("--res", default="720p", choices=sorted(RESOLUTIONS))
    ap.add_argument("--vae-tiles", type=int, default=None, help="VAE decode strips (default 1 @720p, 2 @1080p)")
    ap.add_argument("--clip-retries", type=int, default=3, help="per-clip retries after CUDA OOM")
    ap.add_argument("--derive-720p", choices=["auto", "yes", "no"], default="auto",
                    help="with --res 1080p also write the 720p twin by downscaling the same frames (auto: on for gen3)")
    args = ap.parse_args()
    derive = args.res == "1080p" and (args.derive_720p == "yes" or (args.derive_720p == "auto" and args.set == "gen3"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tiers = set(args.tiers.split(","))
    def twin_id(j: dict) -> str:
        return j["id"][: -len(RESOLUTIONS["1080p"]["suffix"])]

    def done(j: dict) -> bool:
        return (out_dir / f"{j['id']}.mkv").exists() and (not derive or (out_dir / f"{twin_id(j)}.mkv").exists())

    def write_meta(meta: dict) -> None:
        (out_dir / f"{meta['id']}.json").write_text(json.dumps(meta, indent=1))
        with open(out_dir / "manifest.jsonl", "a") as f:
            f.write(json.dumps(meta) + "\n")

    def derived_meta(meta: dict) -> dict:
        return {**meta, "id": twin_id({"id": meta["id"]}), "path": str(out_dir / f"{twin_id({'id': meta['id']})}.mkv"),
                "width": 1280, "height": 720, "resolution": "720p",
                "downscaled": "lanczos from the 1080p generation's frames (crop 1920x1080, scale 1280x720)"}

    jobs = [j for j in build_jobs(args.set, args.res) if j["tier"] in tiers and not done(j)]
    if derive:
        # 1080p already generated, only the 720p twin missing: downscale the stored clip (no GPU work)
        for j in [j for j in jobs if (out_dir / f"{j['id']}.mkv").exists()]:
            src, dst = out_dir / f"{j['id']}.mkv", out_dir / f"{twin_id(j)}.mkv"
            tmp = dst.with_suffix(".tmp.mkv")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf",
                            "scale=1280:720:flags=lanczos,format=yuv420p", "-c:v", "ffv1", "-level", "3",
                            "-threads", "4", str(tmp)], check=True)
            tmp.rename(dst)
            write_meta(derived_meta(json.loads((out_dir / f"{j['id']}.json").read_text())))
            print(f"{twin_id(j)}: 720p twin from stored {src.name}", flush=True)
        jobs = [j for j in jobs if not done(j)]
    print(f"{len(jobs)} clips to generate -> {out_dir}" + (" (+ derived 720p twins)" if derive else ""), flush=True)
    if not jobs:
        return

    import torch
    import wan
    from wan.configs import WAN_CONFIGS

    cfg = WAN_CONFIGS["ti2v-5B"]
    res = RESOLUTIONS[args.res]
    gen_w, gen_h = res["gen"]
    out_w, out_h = res["out"]
    model = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=str(CKPT),
        device_id=0,
        rank=0,
        t5_cpu=args.offload,
        convert_model_dtype=True,
    )
    n_tiles = args.vae_tiles if args.vae_tiles else (2 if args.res == "1080p" else 1)
    if n_tiles > 1:
        model.vae.decode = tiled_vae_decode(model.vae.decode, n_tiles)
    min_free = 36.0 if args.res == "1080p" else 0.0  # DiT sampling at 1920x1088 peaks ~30 GiB

    for j in jobs:
        t0 = time.time()
        video = None
        for attempt in range(args.clip_retries + 1):
            if min_free:
                wait_for_gpu(min_free)
            try:
                video = model.generate(
                    j["prompt"],
                    size=(gen_w, gen_h),
                    max_area=gen_w * gen_h,
                    frame_num=cfg.frame_num,
                    shift=cfg.sample_shift,
                    sample_solver="unipc",
                    sampling_steps=args.steps,
                    guide_scale=cfg.sample_guide_scale,
                    seed=j["seed"],
                    offload_model=True,
                )
                break
            except torch.OutOfMemoryError as exc:
                print(f"{j['id']}: CUDA OOM (attempt {attempt + 1}): {str(exc)[:160]}", flush=True)
                model.model.cpu()
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(60)
        if video is None:
            print(f"{j['id']}: skipped after {args.clip_retries + 1} OOM attempts", flush=True)
            continue
        out = out_dir / f"{j['id']}.mkv"
        write_ffv1(video, out, res["vf"], fps=j["fps"])
        if derive:
            write_ffv1(video, out_dir / f"{twin_id(j)}.mkv", DERIVED_720P_VF, fps=j["fps"])
        del video
        torch.cuda.empty_cache()
        meta = {
            "id": j["id"], "kind": "ai",
            "style": j["style"] or {"anim": "animated", "animh": "animated_human"}.get(args.set, "live"),
            "intended_tier": j["tier"], "prompt": j["prompt"],
            "seed": j["seed"], "steps": args.steps, "seconds": round(time.time() - t0, 1),
            "source": "Wan2.2-TI2V-5B", "licence": LICENCE, "path": str(out),
            "width": out_w, "height": out_h, "fps": float(j["fps"]), "frames": cfg.frame_num,
            "native_hfr": j["fps"] > 30, "hfr_method": "wan_slow_motion_prompt" if j["fps"] > 30 else None,
            "gen_size": [gen_w, gen_h], "resolution": args.res, "vae_tiles": n_tiles,
        }
        write_meta(meta)
        if derive:
            write_meta(derived_meta(meta))
        print(f"{j['id']}: {meta['seconds']}s" + (f" (+ {twin_id(j)} downscaled)" if derive else ""), flush=True)


if __name__ == "__main__":
    main()
